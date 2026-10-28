"""Browser behavior and layout checks; needs Playwright and Microsoft Edge.

Run from week8: python scripts/check_web.py [--url https://...]
Screenshots and detailed browser output stay in ignored artifacts/.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sampling import distribution


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default=(ROOT.parent/'docs/week8-language-lab.html').as_uri())
    args=parser.parse_args()
    screenshots=ROOT/'artifacts/web-qa';screenshots.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        response=page.goto(args.url,wait_until='networkidle')
        if response is not None:assert response.status==200
        title=page.locator('header h1').bounding_box()
        tabs=page.get_by_role('tablist').bounding_box()
        # A right-hand segmented control shares the header row with the title.
        assert tabs['x']>title['x']+title['width'], 'Switch must be right of title'
        assert tabs['x']+tabs['width']>page.viewport_size['width']-60, 'Switch must reach the header right edge'
        assert abs(tabs['y']+tabs['height']/2-(title['y']+title['height']/2))<65
        assert page.locator('#studio').is_visible()
        assert not page.locator('#evidence').is_visible()
        assert not page.locator('.metrics').is_visible(), 'Computed metrics belong only in Evidence'
        page.select_option('#store','0')
        page.eval_on_selector('#temperature',"e=>{e.value='1.2';e.dispatchEvent(new Event('input'));}")
        page.screenshot(path=str(screenshots/'studio-desktop.png'),full_page=False)
        page.click('#evidence-tab')
        assert not page.locator('#studio').is_visible()
        assert page.locator('#evidence').is_visible() and page.locator('.metrics').is_visible()
        assert page.locator('#bpc').inner_text()=='1.220'
        assert page.locator('#gallery article').count()==14
        page.locator('#gallery article').last.scroll_into_view_if_needed()
        page.wait_for_function("Array.from(document.querySelectorAll('#gallery img')).every(i=>i.complete&&i.naturalWidth>0)")
        page.locator('#gallery button').first.click()
        assert page.locator('#lightbox').is_visible()
        page.keyboard.press('Escape')
        assert not page.locator('#lightbox').is_visible()
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(screenshots/'evidence-desktop.png'),full_page=False)
        page.locator('#evidence-tab').focus();page.keyboard.press('ArrowLeft')
        assert page.locator('#studio-tab').get_attribute('aria-selected')=='true'
        assert page.locator('#studio-tab').evaluate('e=>e===document.activeElement')
        assert page.locator('#store').input_value()=='0'
        assert page.locator('#temperature').input_value()=='1.2'
        page.keyboard.press('End')
        assert page.locator('#evidence-tab').get_attribute('aria-selected')=='true'
        page.keyboard.press('Home')
        assert page.locator('#studio-tab').get_attribute('aria-selected')=='true'
        page.keyboard.press('ArrowLeft')
        assert page.locator('#evidence-tab').get_attribute('aria-selected')=='true'
        page.keyboard.press('ArrowRight')
        assert page.locator('#studio-tab').get_attribute('aria-selected')=='true'
        differences=[]
        for context in page.evaluate('DATA.contexts'):
            for mode,settings in [('temperature',{}),('topk',{'top_k':20}),('topp',{'top_p':.9}),('minp',{'min_p':.1})]:
                for temperature in [.1,.7,1.,1.8]:
                    actual=page.evaluate('v=>samplingDistribution(v.logits,v.t,v.mode)',{'logits':context['logits'],'t':temperature,'mode':mode})
                    expected=distribution(context['logits'],temperature=temperature,**settings)
                    differences.append(float(np.max(abs(np.array(actual)-expected))))
        assert max(differences)<1e-12
        for width in [390,700,1024]:
            page.set_viewport_size({'width':width,'height':844})
            for tab,panel in [('evidence-tab','evidence'),('studio-tab','studio')]:
                page.click('#'+tab);assert page.locator('#'+panel).is_visible()
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),f'Overflow in {panel} at {width}px'
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(screenshots/f'studio-{width}.png'),full_page=False)
        assert not errors,errors
        result={'browser':'Microsoft Edge, Playwright headless','desktop':[1440,1000],
                'responsive_widths':[390,700,1024],'figures_loaded':14,'page_errors':errors,
                'sampler_comparisons':len(differences),'max_probability_error':max(differences),
                'checks':['upper-right segmented switch','mutually exclusive panels',
                          'computed metrics only in Evidence','control state preserved',
                          'arrow/Home/End tab navigation','no horizontal overflow',
                          '14 evidence images loaded','lightbox open and Escape close']}
        (ROOT/'evidence/web-validation.json').write_text(json.dumps(result,indent=2)+'\n')
        browser.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

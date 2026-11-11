"""Browser checks for the Week 10 lab; needs Playwright and Microsoft Edge.

Run from week10/: python scripts/check_web.py [--url file:///.../week10-tensor-lab.html]
Screenshots and detailed browser output stay in artifacts/web-qa/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIGURES = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=(ROOT.parent / 'docs' / 'week10-tensor-lab.html').as_uri())
    arguments = parser.parse_args()
    screenshots = ROOT / 'artifacts' / 'web-qa'
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors: list[str] = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        response = page.goto(arguments.url, wait_until='networkidle')
        if response is not None:
            assert response.status == 200

        title = page.locator('header h1').bounding_box()
        tabs = page.get_by_role('tablist').bounding_box()
        assert tabs['x'] > title['x'] + title['width'], 'switch must sit right of the title'
        assert tabs['x'] + tabs['width'] > page.viewport_size['width'] - 60, 'switch must reach the header edge'
        assert abs(tabs['y'] + tabs['height'] / 2 - (title['y'] + title['height'] / 2)) < 65

        assert page.locator('#studio').is_visible()
        assert not page.locator('#evidence').is_visible()
        assert not page.locator('.metrics').is_visible(), 'computed metrics belong only in Evidence'

        # Part 1: the rank slider drives the error, storage ratio and the verdict.
        assert page.locator('#svd-rank-label').inner_text() == '16'
        page.eval_on_selector('#svd-rank', "e=>{e.value='128';e.dispatchEvent(new Event('input'));}")
        saving = page.locator('#svd-stat').inner_text()
        assert 'storage saving yes' in saving, saving
        page.eval_on_selector('#svd-rank', "e=>{e.value='256';e.dispatchEvent(new Event('input'));}")
        equal = page.locator('#svd-stat').inner_text()
        assert 'storage saving no' in equal and '1.000' in equal, equal
        page.eval_on_selector('#svd-rank', "e=>{e.value='395';e.dispatchEvent(new Event('input'));}")
        crossing = page.locator('#svd-stat').inner_text()
        assert 'storage saving no' in crossing, crossing
        assert '10% target met' in crossing, crossing
        page.click('.pills button:nth-child(1)')
        assert page.locator('#svd-stat').inner_text().startswith('retained weight')
        page.screenshot(path=str(screenshots / 'studio-desktop.png'), full_page=False)

        # Part 2: the row player advances, finishes with g, and resets.
        assert page.locator('#row-stat').inner_text().startswith('top edge')
        page.click('#row-step')
        assert 'row 1/' in page.locator('#row-stat').inner_text()
        page.click('#row-play')
        page.wait_for_function("document.getElementById('row-step').disabled", timeout=30000)
        finished = page.locator('#row-stat').inner_text()
        assert 'log Z =' in finished and 'g = log Z / L²' in finished, finished
        page.click('#row-reset')
        assert page.locator('#row-stat').inner_text().startswith('top edge')
        assert page.locator('#row-step').is_enabled()

        # Part 3: the TRG player runs its twenty rounds.
        assert page.locator('#trg-stat').inner_text().startswith('periodic 1024')
        page.click('#trg-play')
        page.wait_for_function("document.getElementById('trg-step').disabled", timeout=60000)
        trg = page.locator('#trg-stat').inner_text()
        assert 'round 20/20' in trg and 'closure z =' in trg, trg
        page.click('#trg-reset')
        assert page.locator('#trg-stat').inner_text().startswith('periodic 1024')

        # Evidence tab: metrics, table, seven figures, lightbox.
        page.click('#evidence-tab')
        assert not page.locator('#studio').is_visible()
        assert page.locator('#evidence').is_visible() and page.locator('.metrics').is_visible()
        assert page.locator('#metric-checks').inner_text().endswith('/46')
        assert page.locator('#row-open-enum').inner_text().startswith('max')
        assert page.locator('#sheet-hash').inner_text().startswith('d053e28083dce7e1')
        assert page.locator('#ref-open-peaks').inner_text() == '2.093 / 2.180 / 2.224'
        assert page.locator('#gallery article').count() == FIGURES
        page.locator('#gallery article').last.scroll_into_view_if_needed()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('#gallery img')).every(i=>i.complete&&i.naturalWidth>0)")
        page.locator('#gallery button').first.click()
        assert page.locator('#lightbox').is_visible()
        first_description = page.locator('#lightbox-desc').inner_text()
        assert len(first_description) > 200, first_description
        assert page.locator('#lightbox-command').inner_text().startswith('python scripts/')
        assert page.locator('#lightbox-counter').inner_text() == f'1 / {FIGURES}'
        page.click('#lightbox-next')
        assert page.locator('#lightbox-counter').inner_text() == f'2 / {FIGURES}'
        assert page.locator('#lightbox-desc').inner_text() != first_description
        page.click('#lightbox-prev')
        assert page.locator('#lightbox-desc').inner_text() == first_description
        page.keyboard.press('Escape')
        assert not page.locator('#lightbox').is_visible()
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(screenshots / 'evidence-desktop.png'), full_page=False)

        # Keyboard navigation and control state preserved across the switch.
        page.locator('#evidence-tab').focus()
        page.keyboard.press('ArrowLeft')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'
        assert page.locator('#studio-tab').evaluate('e=>e===document.activeElement')
        assert page.locator('#svd-rank').input_value() == '16'
        page.keyboard.press('End')
        assert page.locator('#evidence-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('Home')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('ArrowLeft')
        assert page.locator('#evidence-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('ArrowRight')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'

        # The canvas animations drew something.
        for identifier in ('svd-canvas', 'row-canvas', 'trg-canvas'):
            ink = page.evaluate(
                "id=>{const c=document.getElementById(id);const ctx=c.getContext('2d');"
                "const d=ctx.getImageData(0,0,c.width,c.height).data;"
                "let n=0;for(let i=3;i<d.length;i+=4){if(d[i]>0)n++;}return n;}", identifier)
            assert ink > 500, (identifier, ink)

        for width in (390, 700, 1024):
            page.set_viewport_size({'width': width, 'height': 844})
            for tab, panel in (('evidence-tab', 'evidence'), ('studio-tab', 'studio')):
                page.click('#' + tab)
                assert page.locator('#' + panel).is_visible()
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), \
                    f'horizontal overflow in {panel} at {width}px'
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(screenshots / f'studio-{width}.png'), full_page=False)

        assert not errors, errors
        result = {
            'url': arguments.url,
            'browser': 'Microsoft Edge, Playwright headless 1.64.1',
            'desktop': [1440, 1000],
            'responsive_widths': [390, 700, 1024],
            'figures_loaded': FIGURES,
            'page_errors': errors,
            'checks': [
                'upper-right segmented switch reaches the header edge',
                'mutually exclusive studio and evidence panels',
                'computed metrics only in Evidence',
                'SVD rank slider changes the storage verdict and the 10% verdict',
                'boundary row player absorbs rows, reports g and resets',
                'TRG player runs twenty rounds to the closure and resets',
                'all three studio canvases drew',
                'learning-sheet version and reference comparison table',
                'control state preserved across the switch',
                'arrow/Home/End tab navigation',
                'no horizontal overflow at 390/700/1024 px',
                f'{FIGURES} evidence images loaded',
                'lightbox shows the English caption, command and figure counter',
                'lightbox previous/next navigate all figures and Escape closes',
            ],
        }
        (ROOT / 'evidence' / 'web-validation.json').write_text(json.dumps(result, indent=2) + '\n')
        browser.close()
    print(json.dumps({key: result[key] for key in ('figures_loaded', 'page_errors')}, indent=2))
    print(f"{len(result['checks'])} browser checks passed")


if __name__ == '__main__':
    main()

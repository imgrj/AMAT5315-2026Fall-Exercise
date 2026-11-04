"""Browser checks for the Week 9 lab; needs Playwright and Microsoft Edge.

Run from week9/: python scripts/check_web.py [--url file:///.../week9-inference-lab.html]
Screenshots and detailed browser output stay in artifacts/web-qa/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=(ROOT.parent / 'docs' / 'week9-inference-lab.html').as_uri())
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

        # Part 2: the disease bars follow the selected observations.
        prior_text = page.locator('#disease-bars').inner_text()
        page.select_option('#case', 'ADX')
        adx_text = page.locator('#disease-bars').inner_text()
        assert prior_text != adx_text, 'bars must follow the case selector'
        assert '0.3917' in adx_text or '0.391' in adx_text, adx_text

        # Part 3: stepping changes the plan state, and the state survives a tab switch.
        page.select_option('#order', 'greedy')
        assert page.locator('#step-detail').inner_text().startswith('Press Step')
        page.click('#step')
        first_step = page.locator('#step-detail').inner_text()
        assert first_step.startswith('step 1/'), first_step
        page.click('#step')
        assert page.locator('#order-stat').inner_text().startswith('2/'), page.locator('#order-stat').inner_text()
        page.screenshot(path=str(screenshots / 'studio-desktop.png'), full_page=False)

        # Part 4: moving the sweep slider changes the recorded state.
        page.eval_on_selector('#bp-sweep', "e=>{e.value='20';e.dispatchEvent(new Event('input'));}")
        assert 'sweep 20' in page.locator('#bp-stat').inner_text()
        beliefs = page.locator('#belief-bars').inner_text()
        assert 'T belief' in beliefs and 'L belief' in beliefs, beliefs
        page.click('#bp-play')
        page.wait_for_timeout(400)
        page.click('#bp-play')

        # Evidence tab: metrics, table, eight figures, lightbox.
        page.click('#evidence-tab')
        assert not page.locator('#studio').is_visible()
        assert page.locator('#evidence').is_visible() and page.locator('.metrics').is_visible()
        assert page.locator('#metric-mass').inner_text()
        assert page.locator('#row-plans').inner_text().startswith('greedy work')
        assert page.locator('#gallery article').count() == 8
        page.locator('#gallery article').last.scroll_into_view_if_needed()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('#gallery img')).every(i=>i.complete&&i.naturalWidth>0)")
        page.locator('#gallery button').first.click()
        assert page.locator('#lightbox').is_visible()
        page.keyboard.press('Escape')
        assert not page.locator('#lightbox').is_visible()
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(screenshots / 'evidence-desktop.png'), full_page=False)

        # Keyboard navigation and preserved control state.
        page.locator('#evidence-tab').focus()
        page.keyboard.press('ArrowLeft')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'
        assert page.locator('#studio-tab').evaluate('e=>e===document.activeElement')
        assert page.locator('#case').input_value() == 'ADX'
        assert page.locator('#order').input_value() == 'greedy'
        assert page.locator('#order-stat').inner_text().startswith('2/')
        page.keyboard.press('End')
        assert page.locator('#evidence-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('Home')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('ArrowLeft')
        assert page.locator('#evidence-tab').get_attribute('aria-selected') == 'true'
        page.keyboard.press('ArrowRight')
        assert page.locator('#studio-tab').get_attribute('aria-selected') == 'true'

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
            'browser': 'Microsoft Edge, Playwright headless',
            'desktop': [1440, 1000],
            'responsive_widths': [390, 700, 1024],
            'figures_loaded': 8,
            'page_errors': errors,
            'checks': [
                'upper-right segmented switch',
                'mutually exclusive studio and evidence panels',
                'computed metrics only in Evidence',
                'disease bars follow the observation selector',
                'contraction player advances one pair step at a time',
                'BP sweep slider follows the recorded traces',
                'control state preserved across the switch',
                'arrow/Home/End tab navigation',
                'no horizontal overflow at 390/700/1024 px',
                '8 evidence images loaded',
                'lightbox open and Escape close',
            ],
        }
        (ROOT / 'evidence' / 'web-validation.json').write_text(json.dumps(result, indent=2) + '\n')
        browser.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

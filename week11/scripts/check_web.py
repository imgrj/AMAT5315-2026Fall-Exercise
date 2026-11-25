"""Browser checks for the Week 11 lab; needs Playwright and Microsoft Edge.

Run from week11/: python scripts/check_web.py [--url file:///.../week11-tnsample-lab.html]

Screenshots and the check list stay in artifacts/web-qa/ and
evidence/web-validation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIGURES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=(ROOT.parent / 'docs' / 'week11-tnsample-lab.html').as_uri())
    arguments = parser.parse_args()
    screenshots = ROOT / 'artifacts' / 'web-qa'
    screenshots.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors: list[str] = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        response = page.goto(arguments.url, wait_until='networkidle')
        if response is not None:
            assert response.status == 200, response.status

        # The switch sits at the header's right edge, as in Weeks 5-10.
        title = page.locator('header h1').bounding_box()
        tabs = page.get_by_role('tablist').bounding_box()
        assert tabs['x'] > title['x'] + title['width'], 'switch must sit right of the title'
        assert tabs['x'] + tabs['width'] > page.viewport_size['width'] - 60, 'switch must reach the edge'
        checks.append('upper-right segmented switch reaches the header edge')

        assert page.locator('#studio').is_visible() and not page.locator('#evidence').is_visible()
        checks.append('mutually exclusive studio and evidence panels')

        # Panel 1: the gate ladder follows the chips and the player.
        assert page.locator('#gateName').inner_text() == 'AND'
        assert page.locator('#gateFloor').inner_text() == '-3'
        page.click('#gateChips button[data-gate="XOR"]')
        assert page.locator('#gateFloor').inner_text() == '-4', page.locator('#gateFloor').inner_text()
        assert '4 of 8' in page.locator('#gateAllowed').inner_text()
        page.click('#gatesPlay')
        page.wait_for_timeout(900 * 3)
        played = page.locator('#gateName').inner_text()
        assert played != 'AND', played
        page.click('#gatesReset')
        assert page.locator('#gateName').inner_text() == 'AND'
        checks.append('gate chips, player and reset drive the ladder')

        # Panel 2: the contraction runs to the end and lands on (E_min, g0).
        page.click('#contractPlay')
        page.wait_for_timeout(6000)
        assert page.locator('#tropResult').inner_text() == '-299 / 1', page.locator('#tropResult').inner_text()
        assert page.locator('#nodeMax').inner_text() == '32', page.locator('#nodeMax').inner_text()
        page.screenshot(path=str(screenshots / 'studio-desktop.png'), full_page=False)
        checks.append('contraction player reaches E_min = -299 with g_0 = 1')

        # Panel 3: the draws stream and report the Boltzmann comparison.
        page.click('#samplePlay')
        page.wait_for_timeout(600)
        assert int(page.locator('#drawCount').inner_text()) > 0
        page.wait_for_timeout(5000)
        assert page.locator('#drawCount').inner_text().strip() == '2000'
        note = page.locator('#sampleNote').inner_text()
        assert '1455.29' in note, note
        checks.append('draw player streams 2000 samples and quotes the prediction')

        # Panel 4: the board player walks N-queens boards.
        assert page.locator('#queenCount').inner_text() == '4'
        page.click('#queenSizes button[data-n="8"]')
        assert page.locator('#queenCount').inner_text() == '92'
        page.click('#queenStep')
        assert page.locator('#queenDrawn').inner_text().startswith('2 /')
        page.click('#queenSizes button[data-n="6"]')
        page.click('#queenPlay')
        page.wait_for_timeout(1200)
        assert int(page.locator('#queenDrawn').inner_text().split(' / ')[0]) > 3
        checks.append('N-queens player counts boards and steps draws for N = 6 and 8')

        # The evidence tab.
        page.click('#tabEvidence')
        page.wait_for_timeout(200)
        assert page.locator('#evidence').is_visible() and not page.locator('#studio').is_visible()
        assert page.locator('#acceptanceTable tbody tr').count() == 52
        assert page.locator('#predTable tbody tr').count() == 3
        assert page.locator('#scaleTable tbody tr').count() == 7
        checks.append('acceptance, pred and scaling tables are filled')

        images = page.locator('figure.fig img')
        assert images.count() == FIGURES
        for index in range(FIGURES):
            assert page.evaluate('(image) => image.complete && image.naturalWidth > 0',
                                 images.nth(index).element_handle())
        checks.append(f'{FIGURES} evidence images loaded')

        images.nth(1).click()
        assert page.locator('#lbCaption').inner_text().strip() != ''
        assert 'Command:' in page.locator('#lbCommand').inner_text()
        assert page.locator('#lbCount').inner_text() == '2 / 3'
        page.keyboard.press('ArrowRight')
        assert page.locator('#lbCount').inner_text() == '3 / 3'
        page.keyboard.press('Escape')
        assert not page.locator('#lightbox').is_visible()
        checks.append('lightbox shows captions, commands, next/prev and Escape')

        page.screenshot(path=str(screenshots / 'evidence-desktop.png'), full_page=False)

        # Responsive: nothing may overflow horizontally, and no script may fail.
        for width, height in ((1024, 900), (700, 900), (390, 844)):
            page.set_viewport_size({'width': width, 'height': height})
            page.click('#tabStudio')
            page.wait_for_timeout(250)
            page.screenshot(path=str(screenshots / f'studio-{width}.png'), full_page=False)
            overflow = page.evaluate('() => document.documentElement.scrollWidth - '
                                     'document.documentElement.clientWidth')
            assert overflow <= 1, (width, overflow)
            boxes = page.locator('.tab-buttons').bounding_box()
            assert boxes['x'] + boxes['width'] <= width + 1
        checks.append('no horizontal overflow at 390/700/1024 px')
        assert not errors, errors
        checks.append('no page errors')

        browser.close()

    record = {
        'url': arguments.url,
        'browser': 'Microsoft Edge, Playwright headless',
        'desktop': [1440, 1000],
        'responsive_widths': [390, 700, 1024],
        'figures_loaded': FIGURES,
        'page_errors': errors,
        'checks': checks,
    }
    (evidence / 'web-validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'{len(checks)} browser checks passed; screenshots in {screenshots}')


if __name__ == '__main__':
    main()

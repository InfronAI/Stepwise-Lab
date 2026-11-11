"""Exercise report behavior, desktop/mobile layouts and offline resource loading."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.goto((ROOT/'experiments/m1-local/report.html').as_uri())
        assert page.locator('#problemSelect option').count()==32
        before=page.locator('#problem').inner_text()
        page.select_option('#problemSelect','1')
        assert page.locator('#problem').inner_text()!=before
        page.locator('#candidates button').nth(1).click()
        assert page.locator('#candidates button.active').inner_text()=='Candidate 2'
        assert page.locator('.step').count()>0
        assert page.locator('#answer').inner_text().startswith('Final answer:')
        assert page.evaluate('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        browser.close()
    print(json.dumps({'report':'passed','problems':32,'candidate_switch':True,'mobile_no_overflow':True,'javascript_errors':errors}))

if __name__=='__main__':
    main()

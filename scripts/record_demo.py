"""Record an actual CLI run followed by real report interactions (no simulated metrics)."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/media')
    parser.add_argument('--browser', default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    parser.add_argument('--run-dir', default='experiments/runs/video-demo')
    args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run=(ROOT/args.run_dir).resolve()
    command=[sys.executable,'-m','prm800k','demo','--device','mps','--output',str(run)]
    log=output/'demo-terminal.log'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.browser, headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000}, device_scale_factor=1,
                                    record_video_dir=str(output/'raw'), record_video_size={'width':1440,'height':1000})
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.set_content('''<html><style>body{background:#10283f;color:#d8e5f3;padding:70px;font:19px/1.8 monospace}h1{font:36px -apple-system,sans-serif;color:white}p{color:#74d9cc;font:17px -apple-system,sans-serif}pre{white-space:pre-wrap;font:17px/1.9 monospace}</style><p>PRM800K / M1 MAC / LIVE EXECUTION</p><h1>Process supervision, end to end.</h1><pre id="terminal"></pre></html>''')
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            start=time.monotonic()
            try:
                while proc.poll() is None:
                    text='$ python -m prm800k demo --device mps\n\n'+log.read_text()
                    page.locator('#terminal').text_content() # keep browser responsive
                    page.evaluate('(text)=>document.getElementById("terminal").textContent=text',text)
                    page.wait_for_timeout(400)
                    if time.monotonic()-start>600:
                        raise TimeoutError('Demo exceeded 10 minutes')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=15)
        if proc.returncode:
            raise RuntimeError(log.read_text())
        page.evaluate('(text)=>document.getElementById("terminal").textContent=text', '$ python -m prm800k demo --device mps\n\n'+log.read_text())
        page.wait_for_timeout(2000)
        page.goto((run/'report.html').as_uri())
        page.wait_for_load_state('networkidle')
        page.screenshot(path=str(output/'demo-preview.png'), full_page=False)
        page.wait_for_timeout(3000)
        page.locator('#inspect').scroll_into_view_if_needed()
        page.wait_for_timeout(2000)
        page.select_option('#problemSelect','1')
        page.wait_for_timeout(1800)
        page.locator('#candidates button').nth(1).click()
        page.wait_for_timeout(2000)
        page.locator('#training').scroll_into_view_if_needed()
        page.wait_for_timeout(2000)
        page.locator('#overview').scroll_into_view_if_needed()
        page.wait_for_timeout(1800)
        if errors:
            raise RuntimeError(errors)
        video=page.video
        context.close()
        raw=video.path()
        browser.close()
    subprocess.run(['ffmpeg','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-crf','25','-movflags','+faststart',str(output/'demo.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('Recorded:',output/'demo.mp4')

if __name__=='__main__':
    main()

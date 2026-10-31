// render_pdf.js -- print a local HTML file to PDF with headless Chromium.
// Usage: node render_pdf.js input.html output.pdf
const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const [input, output] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(input), { waitUntil: 'load' });
  await page.waitForTimeout(1500);
  await page.pdf({
    path: output,
    format: 'A4',
    printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;"><span class="pageNumber"></span>/<span class="totalPages"></span></div>',
  });
  await browser.close();
})();

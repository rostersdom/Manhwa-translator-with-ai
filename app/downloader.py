import os
import base64


def download_with_playwright(url: str, dest_dir: str) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  playwright not installed, skipping browser download")
        return []

    image_paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-web-security", "--disable-features=IsolateOrigins,site-per-process", "--no-sandbox"],
        )
        page = browser.new_page()
        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

        try:
            page.goto(url, wait_until="load", timeout=30000)
        except Exception as e:
            print(f"  Navigation error: {e}")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        last_height = page.evaluate("document.body.scrollHeight")
        while True:
            page.keyboard.press("PageDown")
            page.wait_for_timeout(500)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        try:
            page.wait_for_selector(".readercontent", timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        data_urls = page.evaluate("""
            () => {
                const results = [];
                function isMangaImg(el) {
                    const src = (el.getAttribute('src') || '').toLowerCase();
                    const alt = (el.getAttribute('alt') || '').toLowerCase();
                    if (src.includes('gravatar') || src.includes('avatar')) return false;
                    if (src.includes('cdn.kingofshojo.com/king-bucket/')) return true;
                    if (src.includes('i.ibb.co')) return true;
                    if (alt.startsWith('ch ') && el.naturalWidth > 200) return true;
                    return false;
                }
                let imgs = [];
                const reader = document.querySelector('.readercontent, .chapterbody, #readerpage, .reading-content');
                if (reader) imgs = Array.from(reader.querySelectorAll('img')).filter(isMangaImg);
                if (imgs.length === 0) imgs = Array.from(document.querySelectorAll('img')).filter(isMangaImg);
                imgs.forEach(img => {
                    try {
                        const canvas = document.createElement('canvas');
                        canvas.width = img.naturalWidth || img.width;
                        canvas.height = img.naturalHeight || img.height;
                        if (canvas.width < 100 || canvas.height < 100) return;
                        canvas.getContext('2d').drawImage(img, 0, 0);
                        results.push(canvas.toDataURL('image/jpeg', 0.92));
                    } catch(e) {}
                });
                return results;
            }
        """)

        for i, data_url in enumerate(data_urls):
            try:
                header, encoded = data_url.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                dest = os.path.join(dest_dir, f"page_{i:03d}.jpg")
                with open(dest, "wb") as f:
                    f.write(img_bytes)
                image_paths.append(dest)
            except Exception as e:
                print(f"  Failed to save image {i}: {e}")

        browser.close()

    return image_paths

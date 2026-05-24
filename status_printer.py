import time
from colorama import init, Fore, Back, Style

init(autoreset=True)

class Status:
    def __init__(self):
        self.start_time = time.time()
        self.page_times = []
        self.total_pages = 0

    def header(self, text: str):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}{Style.RESET_ALL}")

    def step(self, num: int, total: int, text: str):
        print(f"\n{Fore.YELLOW}[{num}/{total}]{Style.RESET_ALL} {Fore.WHITE}{Style.BRIGHT}{text}{Style.RESET_ALL}")

    def ok(self, text: str):
        print(f"  {Fore.GREEN}OK{Style.RESET_ALL}  {text}")

    def info(self, text: str):
        print(f"  {Fore.BLUE}...{Style.RESET_ALL} {text}")

    def warn(self, text: str):
        print(f"  {Fore.YELLOW}WRN{Style.RESET_ALL} {text}")

    def err(self, text: str):
        print(f"  {Fore.RED}ERR{Style.RESET_ALL} {text}")

    def detail(self, text: str):
        print(f"    {Fore.LIGHTBLACK_EX}{text}{Style.RESET_ALL}")

    def timer(self, label: str):
        elapsed = time.time() - self.start_time
        print(f"  {Fore.MAGENTA}TIME{Style.RESET_ALL} {label}: {elapsed:.1f}s")

    def page_progress(self, page: int, total: int):
        pct = (page / total) * 100
        bar_len = 20
        filled = int(bar_len * page / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n{Fore.CYAN}  Page {page}/{total} [{bar}] {pct:.0f}%{Style.RESET_ALL}")

    def divider(self):
        print(f"  {Fore.LIGHTBLACK_EX}{'-'*56}{Style.RESET_ALL}")

    def summary(self, total_time: float, pages: int, detected: int, translated: int, skipped: int, errors: int):
        print(f"\n{Fore.GREEN}{Style.BRIGHT}{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        print(f"  Pages:      {pages}")
        print(f"  Detected:   {detected} text regions")
        print(f"  Translated: {translated}")
        print(f"  Skipped:    {skipped}")
        print(f"  Errors:     {errors}")
        print(f"  Time:       {total_time:.1f}s ({total_time/max(pages,1):.1f}s/page)")
        print(f"{'='*60}\n")

status = Status()

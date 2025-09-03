import requests

def check_internet(url="http://www.google.com", timeout=5):
    """Check general internet connectivity. Returns True/False (no prints when imported)."""
    try:
        # use HEAD for a lighter-weight check
        response = requests.head(url, timeout=timeout)
        return response.status_code < 400
    except requests.RequestException:
        return False

def check_gaia_server(timeout=10):
    """Check if Gaia TAP server is reachable. Returns True/False."""
    gaia_url = "https://gea.esac.esa.int/tap-server/tap"
    try:
        response = requests.head(gaia_url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

if __name__ == "__main__":
    print("Checking general internet connectivity...")
    internet_ok = check_internet()
    print(f"Internet reachable: {internet_ok}")
    print("\nChecking Gaia TAP server connectivity...")
    gaia_ok = check_gaia_server()
    print(f"Gaia TAP reachable: {gaia_ok}")

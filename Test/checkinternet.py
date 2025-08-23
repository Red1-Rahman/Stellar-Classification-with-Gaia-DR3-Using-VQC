import requests

def check_internet(url="http://www.google.com", timeout=5):
    """Check general internet connectivity."""
    try:
        response = requests.get(url, timeout=timeout)
        print(f"Internet is reachable. Status code: {response.status_code}")
        return True
    except requests.ConnectionError:
        print("No internet connection available.")
        return False
    except requests.Timeout:
        print("Internet connection timed out.")
        return False

def check_gaia_server(timeout=10):
    """Check if Gaia TAP server is reachable."""
    gaia_url = "https://gea.esac.esa.int/tap-server/tap"
    try:
        response = requests.get(gaia_url, timeout=timeout)
        if response.status_code == 200:
            print("Gaia TAP server is reachable!")
            return True
        else:
            print(f"Gaia TAP server responded with status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("Cannot connect to Gaia TAP server. Check your internet/firewall.")
        return False
    except requests.Timeout:
        print("Connection to Gaia TAP server timed out.")
        return False

if __name__ == "__main__":
    print("Checking general internet connectivity...")
    check_internet()
    print("\nChecking Gaia TAP server connectivity...")
    check_gaia_server()

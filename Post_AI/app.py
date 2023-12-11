from pypost import create_app
from pyfiglet import Figlet

def initialize():
    f = Figlet(font='slant', width=200)
    ascii_art = f.renderText('Team PostProduction')
    print(ascii_art)

app = create_app()

if __name__ == '__main__':
    initialize()
    app.run(host="0.0.0.0", port=8888, debug=True)
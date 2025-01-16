import gspread

gc = gspread.service_account()
sh = gc.open("landscape")

print(sh.sheet1.get('A1'))
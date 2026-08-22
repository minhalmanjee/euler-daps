import pickle

with open("/home/mmanjee/CARS/processed/lanl/nmap.pkl", "rb") as f:
    nmap = pickle.load(f)

print(len(nmap))                          # 15610
print(len(set(nmap)))                     # same if no duplicate names
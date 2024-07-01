
from scapy.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Capturing ARP packets
def capture_arp_packets(interface, duration=60):
    packets = sniff(filter="arp", iface=interface, timeout=duration)
    wrpcap("arp_packets.pcap", packets)
    print("Packets captured and saved to arp_packets.pcap")

# Feature extraction
def preprocess_arp_packets(pcap_file):
    packets = rdpcap(pcap_file)
    data = []

    for packet in packets:
        if packet.haslayer(ARP):
            src_ip = packet[ARP].psrc
            src_mac = packet[ARP].hwsrc
            dst_ip = packet[ARP].pdst
            dst_mac = packet[ARP].hwdst
            data.append([src_ip, src_mac, dst_ip, dst_mac])
    
    df = pd.DataFrame(data, columns=["SrcIP", "SrcMAC", "DstIP", "DstMAC"])
    return df

#Model training
def train_and_evaluate_model(df):

    df['Pair'] = df['SrcIP'] + '-' + df['SrcMAC']
    df['Count'] = df.groupby('Pair')['Pair'].transform('count')
    df['Label'] = df['Count'].apply(lambda x: 1 if x > 1 else 0)  

    X = df[['SrcIP', 'SrcMAC', 'DstIP', 'DstMAC', 'Count']]
    y = df['Label']

    #Encoding categorical features
    X = pd.get_dummies(X, columns=["SrcIP", "SrcMAC", "DstIP", "DstMAC"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred)
if __name__ == "__main__":
    
    capture_arp_packets(interface="wlan0")

    df = preprocess_arp_packets("arp_packets.pcap")

    train_and_evaluate_model(df)

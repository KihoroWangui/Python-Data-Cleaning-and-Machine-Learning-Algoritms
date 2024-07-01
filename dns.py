
from scapy.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def capture_dns_packets(interface, duration=60):
    packets = sniff(filter="udp port 53", iface=interface, timeout=duration)
    wrpcap("dns_packets.pcap", packets)
    print("Packets captured and saved to dns_packets.pcap")

def preprocess_dns_packets(pcap_file):
    packets = rdpcap(pcap_file)
    data = []
#DNS response
    for packet in packets:
        if packet.haslayer(DNS) and packet[DNS].qr == 1:  
            query_name = packet[DNSQR].qname.decode('utf-8')
            response_ip = packet[DNSRR].rdata if packet[DNS].ancount > 0 else None
            response_time = packet.time
            data.append([query_name, response_ip, response_time])
    
    df = pd.DataFrame(data, columns=["QueryName", "ResponseIP", "ResponseTime"])
    return df

#Model training
def train_and_evaluate_model(df):
    df['ResponseIP'] = df['ResponseIP'].astype(str)
    df['ResponseTimeDiff'] = df['ResponseTime'].diff().fillna(0)
    df['Label'] = df.duplicated(subset=['QueryName', 'ResponseIP']).astype(int) 

    X = df[['QueryName', 'ResponseIP', 'ResponseTimeDiff']]
    y = df['Label']

    X = pd.get_dummies(X, columns=["QueryName", "ResponseIP"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
if __name__ == "__main__":

    capture_dns_packets(interface="wlan0")

    df = preprocess_dns_packets("dns_packets.pcap")

    train_and_evaluate_model(df)

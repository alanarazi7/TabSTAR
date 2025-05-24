from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: KDDCup99
====
Examples: 4898431
====
URL: https://www.openml.org/search?type=data&id=42746
====
Description: INTRUSION DETECTOR LEARNING

Software to detect network intrusions protects a computer network from unauthorized users, including perhaps insiders.  The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between "bad" connections, called intrusions or attacks, and "good" normal connections.
The 1998 DARPA Intrusion Detection Evaluation Program was prepared and managed by MIT Lincoln Labs. The objective was to survey and evaluate research in intrusion detection.  A standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment, was provided.  The 1999 KDD intrusion detection contest uses a version of this dataset.

Lincoln Labs set up an environment to acquire nine weeks of raw TCP dump data for a local-area network (LAN) simulating a typical U.S. Air Force LAN.  They operated the LAN as if it were a true Air Force environment, but peppered it with multiple attacks.

The raw training data was about four gigabytes of compressed binary TCP dump data from seven weeks of network traffic.  This was processed into about five million connection records.  Similarly, the two weeks of test data yielded around two million connection records.

A connection is a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target IP address under some well defined protocol.  Each connection is labeled as either normal, or as an attack, with exactly one specific attack type.  Each connection record consists of about 100 bytes.

Attacks fall into four main categories:

* DOS: denial-of-service, e.g. syn flood;
* R2L: unauthorized access from a remote machine, e.g. guessing password;
* U2R:  unauthorized access to local superuser (root) privileges, e.g., various "buffer overflow" attacks;
* probing: surveillance and other probing, e.g., port scanning.
It is important to note that the test data is not from the same probability distribution as the training data, and it includes specific attack types not in the training data.  This makes the task more realistic.  Some intrusion experts believe that most novel attacks are variants of known attacks and the "signature" of known attacks can be sufficient to catch novel variants.  The datasets contain a total of 24 training attack types, with an additional 14 types in the test data only. 
  
 
DERIVED FEATURES

Stolfo et al. defined higher-level features that help in distinguishing normal connections from attacks.  There are several categories of derived features.
The "same host" features examine only the connections in the past two seconds that have the same destination host as the current connection, and calculate statistics related to protocol behavior, service, etc.

The similar "same service" features examine only the connections in the past two seconds that have the same service as the current connection.

"Same host" and "same service" features are together called  time-based traffic features of the connection records.

Some probing attacks scan the hosts (or ports) using a much larger time interval than two seconds, for example once per minute.  Therefore, connection records were also sorted by destination host, and features were constructed using a window of 100 connections to the same host instead of a time window.  This yields a set of so-called host-based traffic features.

Unlike most of the DOS and probing attacks, there appear to be no sequential patterns that are frequent in records of R2L and U2R attacks. This is because the DOS and probing attacks involve many connections to some host(s) in a very short period of time, but the R2L and U2R attacks are embedded in the data portions 
of packets, and normally involve only a single connection.

Useful algorithms for mining the unstructured data portions of packets automatically are an open research question.  Stolfo et al. used domain knowledge to add features that look for suspicious behavior in the data portions, such as the number of failed login attempts.  These features are called "content" features.

A complete listing of the set of features defined for the connection records is given in the three tables below.  The data schema of the contest dataset is available in machine-readable form . 
  
 

feature namedescription 	type

* duration 	length (number of seconds) of the connection 	continuous
* protocol_type 	type of the protocol, e.g. tcp, udp, etc. 	discrete
* service 	network service on the destination, e.g., http, telnet, etc. 	discrete
* src_bytes 	number of data bytes from source to destination 	continuous
* dst_bytes 	number of data bytes from destination to source 	continuous
* flag 	normal or error status of the connection 	discrete 
* land 	1 if connection is from/to the same host/port; 0 otherwise 	discrete
* wrong_fragment 	number of "wrong" fragments 	continuous
* urgent 	number of urgent packets 	continuous
  
Table 1: Basic features of individual TCP connections.
 
feature namedescription 	type

* hot 	number of "hot" indicatorscontinuous
* num_failed_logins 	number of failed login attempts 	continuous
* logged_in 	1 if successfully logged in; 0 otherwise 	discrete
* num_compromised 	number of "compromised" conditions 	continuous
* root_shell 	1 if root shell is obtained; 0 otherwise 	discrete
* su_attempted 	1 if "su root" command attempted; 0 otherwise 	discrete
* num_root 	number of "root" accesses 	continuous
* num_file_creations 	number of file creation operations 	continuous
* num_shells 	number of shell prompts 	continuous
* num_access_files 	number of operations on access control files 	continuous
* num_outbound_cmdsnumber of outbound commands in an ftp session 	continuous
* is_hot_login 	1 if the login belongs to the "hot" list; 0 otherwise 	discrete
* is_guest_login 	1 if the login is a "guest"login; 0 otherwise 	discrete
  
Table 2: Content features within a connection suggested by domain knowledge.
 
feature namedescription 	type

* count 	number of connections to the same host as the current connection in the past two seconds 	continuous
Note: The following  features refer to these same-host connections.
* serror_rate 	% of connections that have "SYN" errors 	continuous
* rerror_rate 	% of connections that have "REJ" errors 	continuous
* same_srv_rate 	% of connections to the same service 	continuous
* diff_srv_rate 	% of connections to different services 	continuous
* srv_count 	number of connections to the same service as the current connection in the past two seconds 	continuous
Note: The following features refer to these same-service connections.
* srv_serror_rate 	% of connections that have "SYN" errors 	continuous
* srv_rerror_rate 	% of connections that have "REJ" errors 	continuous
* srv_diff_host_rate 	% of connections to different hosts 	continuous 
  
Table 3: Traffic features computed using a two-second time window.
====
Target Variable: target (nominal, 23 distinct): ['smurf.', 'neptune.', 'normal.', 'satan.', 'ipsweep.', 'portsweep.', 'nmap.', 'back.', 'warezclient.', 'teardrop.']
====
Features:

duration (numeric, 9883 distinct): ['0', '1', '2', '3', '5', '2630', '4', '14', '10', '7']
protocol_type (nominal, 3 distinct): ['icmp', 'tcp', 'udp']
service (nominal, 70 distinct): ['ecr_i', 'private', 'http', 'smtp', 'other', 'domain_u', 'ftp_data', 'eco_i', 'finger', 'urp_i']
flag (nominal, 11 distinct): ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1', 'S2', 'RSTOS0', 'OTH']
src_bytes (numeric, 7195 distinct): ['1032', '0', '520', '105', '147', '146', '42', '8', '44', '145']
dst_bytes (numeric, 21493 distinct): ['0', '105', '147', '146', '145', '42', '330', '331', '329', '332']
land (nominal, 2 distinct): ['0', '1']
wrong_fragment (numeric, 3 distinct): ['0', '3', '1']
urgent (numeric, 6 distinct): ['0', '1', '2', '5', '3', '14']
hot (numeric, 30 distinct): ['0', '2', '1', '4', '6', '5', '30', '19', '28', '14']
num_failed_logins (numeric, 6 distinct): ['0', '1', '2', '3', '4', '5']
logged_in (nominal, 2 distinct): ['0', '1']
num_compromised (numeric, 98 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
root_shell (nominal, 2 distinct): ['0', '1']
su_attempted (nominal, 3 distinct): ['0', '2', '1']
num_root (numeric, 93 distinct): ['0', '1', '9', '6', '5', '2', '4', '3', '10', '7']
num_file_creations (numeric, 42 distinct): ['0', '1', '2', '4', '17', '12', '14', '20', '13', '11']
num_shells (numeric, 3 distinct): ['0', '1', '2']
num_access_files (numeric, 10 distinct): ['0', '1', '2', '4', '5', '3', '6', '8', '7', '9']
num_outbound_cmds (numeric, 1 distinct): ['0']
is_host_login (nominal, 2 distinct): ['0', '1']
is_guest_login (nominal, 2 distinct): ['0', '1']
count (numeric, 512 distinct): ['511', '1', '510', '2', '509', '3', '4', '5', '6', '7']
srv_count (numeric, 512 distinct): ['511', '1', '510', '2', '3', '4', '5', '6', '7', '8']
serror_rate (numeric, 96 distinct): ['0.0', '1.0', '0.99', '0.08', '0.05', '0.14', '0.07', '0.06', '0.04', '0.09']
srv_serror_rate (numeric, 87 distinct): ['0.0', '1.0', '0.03', '0.04', '0.05', '0.06', '0.02', '0.08', '0.07', '0.5']
rerror_rate (numeric, 89 distinct): ['0.0', '1.0', '0.86', '0.87', '0.92', '0.95', '0.9', '0.91', '0.93', '0.88']
srv_rerror_rate (numeric, 76 distinct): ['0.0', '1.0', '0.5', '0.33', '0.25', '0.03', '0.2', '0.02', '0.17', '0.12']
same_srv_rate (numeric, 101 distinct): ['1.0', '0.06', '0.05', '0.07', '0.04', '0.03', '0.02', '0.01', '0.08', '0.09']
diff_srv_rate (numeric, 95 distinct): ['0.0', '0.06', '0.07', '0.05', '0.08', '1.0', '0.04', '0.67', '0.5', '0.09']
srv_diff_host_rate (numeric, 72 distinct): ['0.0', '1.0', '0.12', '0.5', '0.67', '0.11', '0.33', '0.25', '0.1', '0.14']
dst_host_count (numeric, 256 distinct): ['255', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dst_host_srv_count (numeric, 256 distinct): ['255', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dst_host_same_srv_rate (numeric, 101 distinct): ['1.0', '0.02', '0.04', '0.05', '0.07', '0.01', '0.0', '0.03', '0.06', '0.08']
dst_host_diff_srv_rate (numeric, 101 distinct): ['0.0', '0.07', '0.06', '0.05', '0.08', '0.01', '0.02', '0.09', '0.03', '0.04']
dst_host_same_src_port_rate (numeric, 101 distinct): ['1.0', '0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.5', '0.08']
dst_host_srv_diff_host_rate (numeric, 76 distinct): ['0.0', '0.02', '0.01', '0.04', '0.03', '0.05', '0.06', '0.07', '0.5', '0.08']
dst_host_serror_rate (numeric, 101 distinct): ['0.0', '1.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.08', '0.07', '0.09']
dst_host_srv_serror_rate (numeric, 100 distinct): ['0.0', '1.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.5', '0.07']
dst_host_rerror_rate (numeric, 101 distinct): ['0.0', '1.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.85', '0.87', '0.93']
dst_host_srv_rerror_rate (numeric, 101 distinct): ['0.0', '1.0', '0.01', '0.02', '0.98', '0.04', '0.99', '0.03', '0.05', '0.97']
'''

CONTEXT = "Software for Network Intrusion Detection"
TARGET = CuratedTarget(raw_name="target", new_name="Attack Name", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []

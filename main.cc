// # SPDX-License-Identifier: MIT
// # © 2025 Manish Kumar

/*
    Filename: main.cc
    
    Description:
    -----------
    Simulation of a dumbbell network topology with a single bottleneck router with DCTCP and ECN enabled.

    The simulation includes: 
    - 6*numFlowsMultiplier  : DCTCP (L4S) flows
    - 4*numFlowsMultiplier  : TCP Cubic (Classic) flows
    - 10*numFlowsMultiplier : L4S background flows (Cross-traffic)
    - a TCP bulk send application on the client and a packet sink on the server.
    - FlowMonitorHelper for flow-level stats
    - Per-packet RTT logging
    - Tracking the following details per packet: Timestamp, PacketSize, five‐tuple FlowID, RTT, ECN marking.

    Usage:
    ------
    - Compile with ns-3: ./ns3 build
    - Run the simulation: ./ns3 run "scratch/main.cc --simNum=2 --bdpMultiplier=4.0 --delayMs=100 --numFlowsMultiplier=1.0"
    - Change parameters as needed using command line arguments.

    Note:
    -----
    - simNum: Simulation number for output directory
    - bdpMultiplier: Multiplier for queue size based on Bandwidth-Delay Product (BDP)
    - delayMs: Propagation delay in milliseconds
    - numFlowsMultiplier: Multiplier for number of flows (affects total flow counts)
    
    - The script uses DualQCoupledPi2QueueDisc for queue management.
    - The queue size is dynamically calculated based on the Bandwidth-Delay Product (BDP).
    - The script ONLY logs packet transmission events with timestamps.

    Contact:
    -------
    manish.kumar.iitd.cse@gmail.com
*/

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/dualq-coupled-pi2-queue-disc.h"
#include "ns3/tcp-dctcp.h"
#include "ns3/tcp-cubic.h" 
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/netanim-module.h"
#include "ns3/timestamp-tag.h"
#include <fstream>
#include <filesystem>

using namespace ns3;
namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("SwiftQueueDumbbell");                          // Enable logging

// GLOBAL parameters
static const int SIM_TIME = 300;                    // Duration : 5 min
static const double STOP_TIME = SIM_TIME + 2.0;     // 2 seconds for graceful shutdown
static const uint32_t TARGET_RATE_Mbps = 5;         // Per-flow rate in Mbps           
static const uint16_t BASE_PORT = 50000;            // Base port for TCP flows

std::ofstream packetTraceFile;                      // for per packet logging
std::unordered_map<uint32_t, Time> packetSendTimes;


// Function to calculate BDP for the given parameters
double CalculateBDP(std::string bandwidth, std::string delay) {
    double bw_bps = 100e6;                          // Parse bandwidth  (e.g., "100Mbps"    -> 100 * 1e6 bits/sec)      & delay
    double delay_sec = std::stod(delay.substr(0, delay.size() - 2)) / 1000.0;                        
    double bdp_bits = bw_bps * delay_sec;           // BDP in bits
    double bdp_packets = bdp_bits / (1500 * 8);     // Convert to packets (assuming 1500 byte packets)

    // Log BDP calculation (in packets)
    NS_LOG_UNCOND("Calculated BDP: " << bdp_packets << " packets");
    return bdp_packets;
}

// Function to calculate the queue size based on BDP
void ConfigureQueueSize(NetDeviceContainer& devices, double bdpMultiplier, double delayMs) {
    std::ostringstream delayStr;
    delayStr << delayMs << "ms";
    double bdp = CalculateBDP("100Mbps", delayStr.str());
    uint32_t totalPackets = static_cast<uint32_t>(bdp * bdpMultiplier);
    uint32_t queueLimitBytes = totalPackets * 1500;                         // Convert packets to bytes (assuming 1500-byte MTU)

    // Queue size configuration
    TrafficControlHelper tch;
    tch.SetRootQueueDisc(
        "ns3::DualQCoupledPi2QueueDisc",
        "QueueLimit", UintegerValue(queueLimitBytes),
        "Target",    TimeValue(MicroSeconds(20000)),
        "Tupdate",   TimeValue(MicroSeconds(1000)),
        "Tshift",    TimeValue(MilliSeconds(50)),
        "A",         DoubleValue(0.1),
        "B",         DoubleValue(0.003),
        "K",         DoubleValue(2.0)
    );
        
    tch.Uninstall(devices.Get(0));                                          // Uninstall previous queue disc on R
    tch.Install(devices.Get(0));                                            // Install DualQCoupledPi2QueueDisc on R's egress device    
}

// Function to calculate send size based on (target rate in Mbps)
uint32_t CalculateSendSize(uint32_t targetRateMbps) {
    uint32_t targetRateBps = targetRateMbps * 1000 * 1000;                  // Convert to bits per second
    uint32_t targetRateBytes = targetRateBps / 8;                           // Convert to bytes per second
    double sendInterval = 0.001;                                            // Send every 1ms
    return (uint32_t)(targetRateBytes * sendInterval);                      // Bytes per interval
}

// Packet tracing function
void PacketTrace(bool isTx, std::string context, Ptr<const Packet> originalPacket) {
    // Extract node and device IDs from context string
    uint32_t nodeId = 0, deviceId = 0;
    {
        auto npos = context.find("/NodeList/");
        if(npos != std::string::npos) {
            nodeId = std::stoi(context.substr(npos + 10));
        }
        
        npos = context.find("/DeviceList/");
        if(npos != std::string::npos) {
            deviceId = std::stoi(context.substr(npos + 12));
        }
    }

    // Copy packet for header parsing
    Ptr<Packet> packet = originalPacket->Copy();
    Time now = Simulator::Now();
    uint32_t packetId = packet->GetUid();
    uint32_t packetSize = packet->GetSize();

    // Default values
    std::string ecnMarking = "NotECT";
    std::string srcIp = "-", dstIp = "-", protocolName = "-";
    uint16_t srcPort = 0, dstPort = 0;
    std::string rtt_us_str = "";

    // Remove PPP header and read IPv4 header
    PppHeader ppp;
    if(packet->RemoveHeader(ppp) && ppp.GetProtocol() == 0x0021) {
        Ipv4Header ipHeader;
        if(packet->RemoveHeader(ipHeader)) {
            // ECN marking
            auto ecn = ipHeader.GetEcn();
            if(ecn == Ipv4Header::ECN_ECT0)           ecnMarking = "L4S";
            else if(ecn == Ipv4Header::ECN_ECT1)      ecnMarking = "Classic";
            else if(ecn == Ipv4Header::ECN_CE)        ecnMarking = "CE";
            else                                      ecnMarking = "NotECT";

            // Get source and destination IP addresses using ostringstream
            std::ostringstream srcStream, dstStream;
            ipHeader.GetSource().Print(srcStream);
            ipHeader.GetDestination().Print(dstStream);
            srcIp = srcStream.str();
            dstIp = dstStream.str();

            // Get protocol number (i.e., TCP = 6, UDP = 17)
            uint8_t proto = ipHeader.GetProtocol();
            protocolName = std::to_string(proto);

            // For TCP, read ports and detect ACK for RTT
            if(proto == 6) {
                TcpHeader tcpHeader;
                if(packet->RemoveHeader(tcpHeader)) {
                    srcPort = tcpHeader.GetSourcePort();
                    dstPort = tcpHeader.GetDestinationPort();
                    
                    if(isTx) {
                        // At TX: record seqNum -> send time
                        uint32_t seqNum = tcpHeader.GetSequenceNumber().GetValue();
                        packetSendTimes[seqNum] = now;
                    } 
                    else if(tcpHeader.GetFlags() & TcpHeader::ACK) {
                        // If this is an ACK on receive side, compute RTT as this packet is tagged with send time
                        uint32_t ackNum = tcpHeader.GetAckNumber().GetValue();
                        auto it = packetSendTimes.find(ackNum);
                        if (it != packetSendTimes.end()) {
                            Time sendTime = it->second;
                            rtt_us_str = std::to_string((now - sendTime).GetMicroSeconds());
                            packetSendTimes.erase(it);              // Optional: free memory
                        }
                    }
                }
            }
        }
    }

    // ReceiverID is same as destination IP
    std::string receiverId = dstIp;

    // Write CSV row
    packetTraceFile
        << now.GetNanoSeconds() << ","
        << nodeId << ","
        << deviceId << ","
        << packetId << ","
        << (isTx ? "TX" : "RX") << ","
        << packetSize << ","
        << ecnMarking << ","
        << srcIp << ","
        << srcPort << ","
        << dstIp << ","
        << dstPort << ","
        << protocolName << ","
        << receiverId << ","
        << rtt_us_str << "\n";
}



// Driver function
int main(int argc, char *argv[]) {
    // CLI params
    int simNum = 0;                                 // Simulation number
    double bdpMultiplier = 2.0;                     // BDP multiplier for queue size    (default: 2.0)
    double delayMs = 20.0;                          // Default delay in ms
    double numFlowsMultiplier = 1.0;                // Multiplier for flow counts

    CommandLine cmd;
    cmd.AddValue("simNum", "Simulation Number", simNum);
    cmd.AddValue("bdpMultiplier", "Queue size multiplier", bdpMultiplier);
    cmd.AddValue("delayMs", "Propagation delay in ms", delayMs);
    cmd.AddValue("numFlowsMultiplier", "Multiplier for number of flows", numFlowsMultiplier);
    cmd.Parse(argc, argv);
    
    // Setup output directory
    std::ostringstream dirNameStream;
    dirNameStream << "Simulations_res/"
                << "bdp" << bdpMultiplier
                << "_delay" << static_cast<int>(delayMs)
                << "_flows" << numFlowsMultiplier
                << "_sim" << simNum << "/";
    std::string outputDir = dirNameStream.str();
    fs::create_directories(outputDir);              // Create output directory if it doesn't exist

    // Initialize packet trace file
    packetTraceFile.open(outputDir + "packet_trace.csv");
    if(!packetTraceFile.is_open()) {
        NS_LOG_UNCOND("Failed to open rtt_per_packet.csv for writing.");
    } else {
        NS_LOG_UNCOND("Logging packet trace to " << outputDir + "packet_trace.csv");
        packetTraceFile << "Timestamp_ns,NodeId,DeviceId,PacketId,Direction,"
                  "PacketSize,ECN_Marking,"
                  "SrcIP,SrcPort,DstIP,DstPort,Protocol,ReceiverID,RTT_us\n";
    }

    // Set random seed for diff. flows
    Ptr<UniformRandomVariable> randStart = CreateObject<UniformRandomVariable>();
    double flowStartWindow = 0.5;       // All flows start within 0.5s window

    // 1. Create Nodes
    NodeContainer clients;
    clients.Create(2);                  // C0 -> sends main traffic, C1 -> send cross traffic

    NodeContainer server;
    server.Create(1);                   // S0

    NodeContainer router;
    router.Create(1);                   // single bottleneck router

    // // Assign constant position mobility model to all nodes
    // MobilityHelper mobility;
    // mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    // mobility.Install(clients);
    // mobility.Install(server);
    // mobility.Install(router);

    // // Set positions for NetAnim visualization
    // clients.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(10, 45, 0));                     // C0
    // clients.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(10, 15, 0));                     // C1
    // router.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(50, 30, 0));                      // R
    // server.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(110, 30, 0));                     // S0


    // 2. Create point-to-point links
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delayMs)));
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1p"));       // Set queue size to 1 packet for bottleneck (to be updated later)

    
    // 3. Connect client and router
    NetDeviceContainer dev_cr0, dev_cr1;
    dev_cr0 = p2p.Install(clients.Get(0), router.Get(0));                   // link C0 -> R
    dev_cr1 = p2p.Install(clients.Get(1), router.Get(0));                   // link C1 -> R


    // 4. Connect router and server
    NetDeviceContainer dev_rs;
    dev_rs = p2p.Install(router.Get(0), server.Get(0));                     // link R -> S0

    
    // 5. Install Internet TCP/IP stack in all nodes (reqd. before assigning IP addresses)
    InternetStackHelper stack;
    stack.InstallAll();

    
    // 6. Assign IP addresses - unique subnets for each link
    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer iface_cr0 = address.Assign(dev_cr0);             // link C0 -> R
    
    address.SetBase("10.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer iface_cr1 = address.Assign(dev_cr1);             // link C1 -> R
    
    address.SetBase("10.0.2.0", "255.255.255.0");
    Ipv4InterfaceContainer iface_rs = address.Assign(dev_rs);               // link R -> S0
    
    // Logging IP addresses
    // NS_LOG_UNCOND("IP Addresses assigned:");
    // NS_LOG_UNCOND("C0 IP          : " << iface_cr0.GetAddress(0));           // C0
    // NS_LOG_UNCOND("R IP (from C0) : " << iface_cr0.GetAddress(1));           // R side of link 1
    // NS_LOG_UNCOND("C1 IP          : " << iface_cr1.GetAddress(0));           // C1
    // NS_LOG_UNCOND("R IP (from C1) : " << iface_cr1.GetAddress(1));           // R side of link 2
    // NS_LOG_UNCOND("R IP (to S)    : " << iface_rs.GetAddress(0));            // R side to S
    // NS_LOG_UNCOND("S0 IP          : " << iface_rs.GetAddress(1));            // S0
    // NS_LOG_UNCOND("");
    

    // 7. Enable DCTCP with ECN (global setting for all TCP sockets)
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));             // Set default TCP type to DCTCP
    Config::SetDefault("ns3::TcpSocketBase::UseEcn", StringValue("On"));                            // Enable ECN globally for all TCP sockets

    // Connect packet trace callbacks for C0 (main traffic sender)
    uint32_t c0_NodeId = clients.Get(0)->GetId();

    std::ostringstream txPath_C0;
    txPath_C0 << "/NodeList/" << c0_NodeId << "/DeviceList/*/$ns3::PointToPointNetDevice/MacTx";            // mark every packet sent by C0 with Timestamp
    Config::Connect(txPath_C0.str(), MakeBoundCallback(&PacketTrace, true));

    std::ostringstream rxPath_C0;
    rxPath_C0 << "/NodeList/" << c0_NodeId << "/DeviceList/*/$ns3::PointToPointNetDevice/MacRx";            // compute RTT for every packet received by C0
    Config::Connect(rxPath_C0.str(), MakeBoundCallback(&PacketTrace, false));


    // 8. Install DualQCoupledPi2QueueDisc queue disc
    NS_LOG_UNCOND("Installing DualQCoupledPi2QueueDisc on router's egress device...");
    ConfigureQueueSize(dev_rs, bdpMultiplier, delayMs);                                                      // Configure queue with BDP-based sizing

    // Logging the devices and nodes
    // NS_LOG_UNCOND("Devices and Nodes:");
    // NS_LOG_UNCOND("Client dev_cr0 : Node " << dev_cr0.Get(0)->GetNode()->GetId());
    // NS_LOG_UNCOND("Router dev_cr0 : Node " << dev_cr0.Get(1)->GetNode()->GetId());
    // NS_LOG_UNCOND("Client dev_cr1 : Node " << dev_cr1.Get(0)->GetNode()->GetId());
    // NS_LOG_UNCOND("Router dev_cr1 : Node " << dev_cr1.Get(1)->GetNode()->GetId());
    // NS_LOG_UNCOND("Router dev_rs  : Node " << dev_rs.Get(0)->GetNode()->GetId());
    // NS_LOG_UNCOND("Server dev_rs  : Node " << dev_rs.Get(1)->GetNode()->GetId());
    // NS_LOG_UNCOND("");


    // 9. Install TCP BulkSend application on sender
    ApplicationContainer allApps, allSinks;
    uint32_t sendSize = CalculateSendSize(TARGET_RATE_Mbps);                  // Rate limiting for flows
    
    // Configure the BulkSendHelper with rate limiting
    // ========== 9A: Main L4S flows from C0 ==========
    uint32_t numL4S = static_cast<uint32_t>(6 * numFlowsMultiplier);
    for(uint32_t i = 0; i < numL4S; ++i) {
        // NS_LOG_UNCOND("L4S Flow " << i << " from C0 to S0 on port " << BASE_PORT + i);
        Address sinkAddress = InetSocketAddress(iface_rs.GetAddress(1), BASE_PORT + i);
        BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
        source.SetAttribute("SendSize", UintegerValue(sendSize));                   // Set send size for rate limiting
        source.SetAttribute("MaxBytes", UintegerValue(0));                          // Unlimited total
        
        ApplicationContainer app = source.Install(clients.Get(0));
        app.Start(Seconds(1.0 + randStart->GetValue(0.0, flowStartWindow)));
        app.Stop(Seconds(STOP_TIME - 0.5));
        allApps.Add(app);

        // Sink
        PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), BASE_PORT + i));
        allSinks.Add(sink.Install(server.Get(0)));
    }

    // ========== 9B: Classic Cubic flows from C0 ==========
    uint32_t numClassic = static_cast<uint32_t>(4 * numFlowsMultiplier);
    for(uint32_t i = 0; i < numClassic; ++i) {
        // Configure TCP Cubic for these specific flows before creating applications
        std::ostringstream pathOss;
        pathOss << "/NodeList/" << clients.Get(0)->GetId() << "/$ns3::TcpL4Protocol/SocketType";            
        Config::Set(pathOss.str(), StringValue("ns3::TcpCubic"));

        // Disable ECN for these flows
        Config::SetDefault("ns3::TcpSocketBase::UseEcn", StringValue("Off"));

        // NS_LOG_UNCOND("Classic Cubic Flow " << i << " from C0 to S0 on port " << BASE_PORT + 200 + i);
        Address sinkAddress = InetSocketAddress(iface_rs.GetAddress(1), BASE_PORT + 200 + i);
        BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
        source.SetAttribute("SendSize", UintegerValue(sendSize));                   // Set send size for rate limiting
        source.SetAttribute("MaxBytes", UintegerValue(0));                          // Unlimited total
        
        ApplicationContainer app = source.Install(clients.Get(0));
        app.Start(Seconds(1.0 + randStart->GetValue(0.0, flowStartWindow)));
        app.Stop(Seconds(STOP_TIME - 0.5));
        allApps.Add(app);

        // Sink
        PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), BASE_PORT + 200 + i));
        allSinks.Add(sink.Install(server.Get(0)));
    }
    
    // Reset ECN for subsequent flows
    Config::SetDefault("ns3::TcpSocketBase::UseEcn", StringValue("On"));

    // ========== 9C: L4S cross-traffic flows from C1 ==========
    uint32_t numCross = static_cast<uint32_t>(10 * numFlowsMultiplier);
    for(uint32_t i = 1; i <= numCross; ++i) {
        // Re-enable ECN for cross-traffic flows
        std::ostringstream pathOss;
        pathOss << "/NodeList/" << clients.Get(1)->GetId() << "/$ns3::TcpL4Protocol/SocketType";            
        Config::Set(pathOss.str(), StringValue("ns3::TcpDctcp"));
        
        // NS_LOG_UNCOND("Background L4S Flow " << i << " from C1 to S0 on port " << BASE_PORT + 300 + i);
        Address sinkAddress = InetSocketAddress(iface_rs.GetAddress(1), BASE_PORT + 300 + i);
        BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
        source.SetAttribute("SendSize", UintegerValue(sendSize));                   // Set send size for rate limiting
        source.SetAttribute("MaxBytes", UintegerValue(0));                          // Unlimited total
        
        ApplicationContainer app = source.Install(clients.Get(1));
        app.Start(Seconds(1.0 + randStart->GetValue(0.0, flowStartWindow)));
        app.Stop(Seconds(STOP_TIME - 0.5));
        allApps.Add(app);

        // Sink
        PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), BASE_PORT + 300 + i));
        allSinks.Add(sink.Install(server.Get(0)));
    }

    NS_LOG_UNCOND("Total applications/flows installed: " << allApps.GetN());

    // Start/Stop all sinks
    allSinks.Start(Seconds(0.0));
    allSinks.Stop(Seconds(STOP_TIME - 0.5));
    

    // 10. Enable Routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    // 11. Enable Flow Monitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();


    // // 12. Enable Tracing
    // AsciiTraceHelper ascii;
    // p2p.EnableAsciiAll(ascii.CreateFileStream(outputDir + "swiftqueue_dumbbell.tr"));
    // p2p.EnablePcapAll(outputDir + "swiftqueue_dumbbell");
    // AnimationInterface anim(outputDir + "swiftqueue_dumbbell.xml");
    // anim.SetStopTime(Seconds(STOP_TIME));


    // 13. Run simulation
    NS_LOG_UNCOND("\n============ Simulation Started ============");
    NS_LOG_UNCOND("Running simulation for " << SIM_TIME << " seconds...");

    Simulator::Stop(Seconds(STOP_TIME));                                 // Buffer to allow sink stop, flow monitor flush
    Simulator::Run();
    
    
    NS_LOG_UNCOND("Saving flow monitor results...");
    packetTraceFile.close();
    monitor->SerializeToXmlFile(outputDir + "flowmonitor-results.xml", true, true);
    NS_LOG_UNCOND("============  Simulation Ended  ============");

    Simulator::Destroy();
    return 0;
}
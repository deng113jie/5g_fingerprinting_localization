1，install omnet++, simu5g and vein
2, replace simu5g with code under simu5g
3, add simulation example with code under veins_nr
4, generate traffic pattern with sumo using osmWebWizard.py
5, To run the simulation example, first start veins
6, and then run the omnet example under veins_nr using the following command:
	opp_run -m -u Cmdenv -n ../../../emulation:../..:../../../src:../../../../inet4.5/examples:../../../../inet4.5/showcases:../../../../inet4.5/src:../../../../inet4.5/tests/validation:../../../../inet4.5/tests/networks:../../../../inet4.5/tutorials:../../../../../veins_inet/src/veins_inet:../../../../../veins_inet/examples/veins_inet:../../../../../veins-veins-5.2/examples/veins:../../../../../veins-veins-5.2/src/veins -l ../../../src/simu5g -l ../../../../inet4.5/src/INET -l ../../../../../veins_inet/src/veins_inet -l ../../../../../veins-veins-5.2/src/veins -c VoIP-DL omnetpp.ini
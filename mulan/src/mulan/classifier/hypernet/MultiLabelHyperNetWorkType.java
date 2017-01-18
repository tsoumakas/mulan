package mulan.classifier.hypernet;



/**
 * Types of Multi-Label Hypernetwork
 * 
 * @author LB
 * @version 2017.1.10
 */
public enum MultiLabelHyperNetWorkType {
	MLHN_GC,  //Multi-Label Hypernetwork for exploiting global label correlation
	MLHN_LC,  //Multi-Label Hypernetwork for exploiting local label correlation
	MLHN_GLC  //Multi-Label Hypernetwork for exploiting global and local label correlation
}

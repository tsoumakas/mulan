package mulan.core.data;

import java.util.Set;

import mulan.core.data.impl.LabelsMetaDataImpl;

/**
 * Represents meta data about label attributes and their structure. 
 * The labels can be organized in hierarchical structure. If no hierarchy 
 * is defined between labels, they are stored in a flat structure. 
 * This means all labels are represented as root {@link LabelNode} element.
 *  
 * @author Jozef Vilcek
 */
public interface LabelsMetaData {

	/**
	 * Gets the unmodifiable {@link Set} of root {@link LabelNode} 
	 * nodes of label attributes hierarchy.
	 * 
	 * @return the {@link Set} of root nodes.
	 */
	Set<LabelNode> getRootLabels();
	
	/**
	 * Gets the {@link LabelNode} specified by label name. The name is unique identifier
	 * of the node and corresponds to label attribute in the data set.
	 * If {@link LabelNode} for given label name does not exists, <code>null</code> is returned.
	 * 
	 * @param labelName the name of label of which the node should be retrieved 
	 * @return the {@link LabelNode} for specified label of <code>null</code>
	 * 		   if {@link LabelNode} does not exists for specified label name
	 */
	LabelNode getLabelNode(String labelName);
	
	/**
	 * Gets the names of all labels. The label name is a unique identifier among all 
	 * labels in the meta data.
	 * 
	 * @return the names of all labels.
	 */
	Set<String> getLabelNames();
	
	/**
	 * Determines if {@link LabelsMetaData} contains a label with specified name.
	 * 
	 * @param labelName the label name
	 * @return 
	 */
	boolean containsLabel(String labelName);
	
	/**
	 * Determines if there is a hierarchy defined between labels. If not, all labels are
	 * represented as root {@link LabelNode} nodes.
	 * 
	 * @return <code>true</code> if there is hierarchy defined between labels; 
	 * 		   <code>false</code> otherwise.
	 */
	boolean IsHierarchy();
	
	/**
	 * Gets the total number of {@link LabelNode} nodes. 
	 * @return 
	 */
	int getNumLabels();
	
	/**
	 * Returns a deep copy of the {@link LabelsMetaDataImpl} instance.
	 * @return
	 */
	LabelsMetaData clone();
}

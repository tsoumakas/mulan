package mulan.core.data;

import java.util.Set;

/**
 * Represents a label attribute as a node in the labels hierarchy.  
 * The identity of the node is represented by label name, which must be unique 
 * within labels hierarchy.
 * 
 * @author Jozef Vilcek
 */
public interface LabelNode {

	/**
	 * Gets the name of the label this node represents.
	 * The name corresponds to the ID of label attribute in the arff data set.
	 * The name of the label must be unique within the data set, because the 
	 * identity of {@link LabelNode} is determined by the name.
	 * 
	 * @return the name of the label this node represents
	 */
	String getName();
	
	/**
	 * Determines whether the {@link LabelNode} has a parent node in a hierarchy. 
	 * 
	 * @return <code>true</code> if the node has parent; <code>false</code> otherwise.
	 */
	boolean hasParent();
	
	/**
	 * Gets the parent {@link LabelNode} of this node if hierarchy exists.
	 * If the node has not a parent {@link LabelNode}, <code>null</code> is returned.
	 * 
	 * @return the parent {@link LabelNode} or <code>null</code> if the parent does not exists.
	 */
	LabelNode getParent();
	
	/**
	 * Determines whether the {@link LabelNode} has child nodes. 
	 * 
	 * @return <code>true</code> if the node has child nodes; <code>false</code> otherwise.
	 */
	boolean hasChildren();
	
	/**
	 * Gets the unmodifiable {@link Set} of child {@link LabelNode} of this node, if hierarchy exists.
	 * If no child nodes exists for this {@link LabelNode}, empty {@link Set} is returned.
	 * 
	 * @return the {@link Set} of child nodes
	 */
	Set<LabelNode> getChildren();
	
}

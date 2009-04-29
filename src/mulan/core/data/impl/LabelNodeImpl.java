package mulan.core.data.impl;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;

import mulan.core.data.LabelNode;

/**
 * Implementation of node representing label attribute and its connection 
 * within a hierarchy of labels. 
 * 
 * @author Jozef Vilcek
 */
@XmlRootElement(name = "label", namespace = LabelsBuilder.LABELS_SCHEMA_NAMESPACE)
@XmlAccessorType(XmlAccessType.NONE)
@XmlType(name = "labelType", propOrder = {"childrenNodes"})
public class LabelNodeImpl implements LabelNode, Serializable {

	private static final long serialVersionUID = -7974176487751728557L;
	
	@XmlAttribute(required = true)
	private final String name;
	@XmlElement(type = LabelNodeImpl.class, name = "label", namespace = LabelsBuilder.LABELS_SCHEMA_NAMESPACE)
	private final Set<LabelNode> childrenNodes;
	private LabelNode parentNode;
	
	/**
	 * Creates a new instance of {@link LabelNodeImpl}.
	 * @param name the name of the label attribute this node represents
	 */
	public LabelNodeImpl(String name){
		this.name = name;
		parentNode = null;
		childrenNodes = new HashSet<LabelNode>();
	}
	
	/**
	 * Empty constructor needs to be defined because of JAXB.
	 * Not intended for use. 
	 */
	@SuppressWarnings("unused")
	private LabelNodeImpl(){
		name = "";
		childrenNodes = new HashSet<LabelNode>();
	}
	
	
	/**
	 * Adds the specified {@link LabelNode} to the set of child nodes. 
	 * The parent of added node is set to reference this {@link LabelNode} instance.
	 * This indicates that there is a hierarchy between these two {@link LabelNode} nodes.
	 * 
	 * @param node the {@link LabelNode} to be removed
	 * @return true if node was actually removed; false node was not in child nodes set
	 * @throws IllegalArgumentException if specified {@link LabelNode} parameter is null
	 */
	public boolean addChildNode(LabelNode node){
		if(node == null){
			throw new IllegalArgumentException("The label node is null.");
		}
		if(!childrenNodes.contains(node)){
			((LabelNodeImpl)node).setParent(this);
		}
		return childrenNodes.add(node);
	}
	
	/**
	 * Removes the specified {@link LabelNode} from the set of child nodes.
	 * The connection between removed {@link LabelNode} and its {@link LabelNode#getParent()}
	 * 
	 * @param node the {@link LabelNode} to be removed
	 * @return true if node was actually removed; false node was not in child nodes set
	 * @throws IllegalArgumentException if specified {@link LabelNode} parameter is null
	 */
	public boolean removeChildNode(LabelNode node){
		if(node == null){
			throw new IllegalArgumentException("The label node is null.");
		}
		if(childrenNodes.contains(node)){
			for(LabelNode item : childrenNodes){
				if(item.equals(node)){
					((LabelNodeImpl)item).setParent(null);
					break;
				}
			}
		}
		return childrenNodes.remove(node);
	}
	
	
	public Set<LabelNode> getChildren() {
		return Collections.unmodifiableSet(childrenNodes);
	}

	public String getName() {
		return name;
	}

	public LabelNode getParent() {
		return parentNode;
	}
	
	protected void setParent(LabelNode node){
		parentNode = node;
	}

	public boolean hasChildren() {
		return !childrenNodes.isEmpty();
	}

	public boolean hasParent() {
		return (parentNode == null) ? false : true;
	}

	
	@Override
	public int hashCode(){
		int hash = 1;
		hash = hash * 31 + name.hashCode();
		return hash;
	}
	
	@Override
	public boolean equals(Object obj){
		
		if(this == obj){
			return true;
		}
		if((obj == null) || (obj.getClass() != this.getClass())){
			return false;
		}

		LabelNodeImpl labelNode = (LabelNodeImpl)obj;
		return name == labelNode.getName();
	}

}

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    LabelNodeImpl.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

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

import mulan.core.ArgumentNullException;

/**
 * Implementation of {@link LabelNode}, representing a label attribute and its connection 
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
    public LabelNodeImpl(String name) {
        if (name == null) {
            throw new ArgumentNullException("name");
        }
        this.name = name;
        parentNode = null;
        childrenNodes = new HashSet<LabelNode>();
    }

    /**
     * Empty constructor needs to be defined because of JAXB.
     * Not intended for use.
     */
    @SuppressWarnings("unused")
    private LabelNodeImpl() {
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
     * @throws ArgumentNullException if specified {@link LabelNode} parameter is null
     * @throws IllegalArgumentException if {@link LabelNode} being added has same name as this {@link LabelNode} instance (parent)
     */
    public boolean addChildNode(LabelNode node) {
        if (node == null) {
            throw new ArgumentNullException("node");
        }
        if (node.getName().equals(name)) {
            throw new IllegalArgumentException("The child label node can not have same name as parent.");
        }
        if (!childrenNodes.contains(node)) {
            ((LabelNodeImpl) node).setParent(this);
        }
        return childrenNodes.add(node);
    }

    /**
     * Removes the specified {@link LabelNode} from the set of child nodes.
     * The connection between removed {@link LabelNode} and its {@link LabelNode#getParent()}
     *
     * @param node the {@link LabelNode} to be removed
     * @return true if node was actually removed; false node was not in child nodes set
     * @throws ArgumentNullException if specified {@link LabelNode} parameter is null
     */
    public boolean removeChildNode(LabelNode node) {
        if (node == null) {
            throw new ArgumentNullException("node");
        }
        if (childrenNodes.contains(node)) {
            for (LabelNode item : childrenNodes) {
                if (item.equals(node)) {
                    ((LabelNodeImpl) item).setParent(null);
                    break;
                }
            }
        }
        return childrenNodes.remove(node);
    }

    /**
     * Gets the children of a label
     *
     * @return a Set of labels
     */
    public Set<String> getChildrenLabels() {
        Set<String> labels = new HashSet<String>();
        for (LabelNode child : childrenNodes) {
            labels.add(child.getName());
        }
        return labels;
    }

    public Set<String> getDescendantLabels() {
        Set<String> labels = new HashSet<String>();
        if (hasChildren()) {
            for (LabelNode child : childrenNodes) {
                labels.addAll(getDescendantLabelsRec(child));
            }
        }
        return labels;
    }

    private Set<String> getDescendantLabelsRec(LabelNode node) {
        Set<String> labels = new HashSet<String>();
        labels.add(node.getName());
        if (node.hasChildren()) {
            for (LabelNode child : node.getChildren()) {
                labels.addAll(getDescendantLabelsRec(child));
            }
        }
        return labels;
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

    /**
     * Sets a node as the parent of this node
     *
     * @param node a node to be set as parent
     */
    protected void setParent(LabelNode node) {
        parentNode = node;
    }

    public boolean hasChildren() {
        return !childrenNodes.isEmpty();
    }

    public boolean hasParent() {
        return (parentNode == null) ? false : true;
    }

    /**
     * The hash code is computed based on label name attribute, which defines the
     * identity of the {@link LabelNodeImpl} node.
     */
    @Override
    public int hashCode() {
        int hash = 1;
        hash = hash * 31 + name.hashCode();
        return hash;
    }

    /**
     * The two {@link LabelNodeImpl} nodes are equal if the are the same
     * (points to the same object) of if they returns same {@link #getName()} value.
     * The name of the labels gives the identity to the {@link LabelNodeImpl}.
     */
    @Override
    public boolean equals(Object obj) {

        if (this == obj) {
            return true;
        }
        if ((obj == null) || (obj.getClass() != this.getClass())) {
            return false;
        }

        LabelNodeImpl labelNode = (LabelNodeImpl) obj;
        return name.equals(labelNode.getName());
    }
}
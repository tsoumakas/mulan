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
 *    LabelsMetaDataImpl.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;

import mulan.core.ArgumentNullException;
import mulan.core.MulanRuntimeException;
import mulan.core.WekaException;
import weka.core.SerializedObject;

/**
 * Implementation of {@link LabelsMetaData} info about labels and their structure. 
 * 
 * @author Jozef Vilcek
 * @see LabelsMetaData
 */
@XmlRootElement(name = "labels", namespace = LabelsBuilder.LABELS_SCHEMA_NAMESPACE)
@XmlAccessorType(XmlAccessType.NONE)
@XmlType(name = "labelsRootType", propOrder = {"rootLabelNodes"})
public class LabelsMetaDataImpl implements LabelsMetaData, Serializable, Externalizable {

    private static final long serialVersionUID = 5098050799557336378L;
    private Map<String, LabelNode> allLabelNodes;
    @XmlElement(type = LabelNodeImpl.class, name = "label", required = true, namespace = LabelsBuilder.LABELS_SCHEMA_NAMESPACE)
    private Set<LabelNode> rootLabelNodes;

    /**
     * Creates a new instance of {@link LabelsMetaDataImpl}.
     */
    public LabelsMetaDataImpl() {
        allLabelNodes = new HashMap<String, LabelNode>();
        rootLabelNodes = new HashSet<LabelNode>();
    }

    /**
     * Adds a root {@link LabelNode}. The node is assumed to has linked all
     * related child nodes, if they exists.
     * The node is added into root set and all child nodes are added into internal mapping.
     * The node is uniquely identified by the label name.
     *
     * @param rootNode the root node with underlying hierarchy of nodes
     * @throws ArgumentNullException if specified root node is null
     */
    public void addRootNode(LabelNode rootNode) {
        if (rootNode == null) {
            throw new ArgumentNullException("rootNode");
        }
        if (rootLabelNodes.contains(rootNode)) {
            throw new IllegalArgumentException(
                    String.format("The root label node '%s' is already added.",
                    rootNode.getName()));
        }

        rootLabelNodes.add(rootNode);
        processNodeIndex(rootNode, IndexingAction.Add);
    }

    public LabelNode getLabelNode(String labelName) {
        return allLabelNodes.get(labelName);
    }

    public Set<String> getLabelNames() {
        return new HashSet<String>(allLabelNodes.keySet());
    }

    public boolean containsLabel(String labelName) {
        return allLabelNodes.containsKey(labelName);
    }

    public boolean isHierarchy() {
        return (allLabelNodes.size() == rootLabelNodes.size()) ? false : true;
    }

    public int getNumLabels() {
        return allLabelNodes.size();
    }

    public Set<LabelNode> getRootLabels() {
        return Collections.unmodifiableSet(rootLabelNodes);
    }

    @SuppressWarnings("unchecked")
    @Override
    public LabelsMetaData clone() {
        Set<LabelNode> rootNodes = null;
        try {
            SerializedObject obj = new SerializedObject(rootLabelNodes);
            rootNodes = (Set<LabelNode>) obj.getObject();
        } catch (Exception ex) {
            throw new WekaException("Failed to create copy of 'root label nodes'.", ex);
        }

        LabelsMetaDataImpl copyResult = new LabelsMetaDataImpl();
        for (LabelNode rootNode : rootNodes) {
            copyResult.addRootNode(rootNode);
        }

        return copyResult;
    }

    /**
     * Removes {@link LabelNode} specified by the name. If there is a hierarchy between
     * label nodes, whole subtree with all its children is also removed.
     *
     * @param labelName the name of {@link LabelNode} to be removed
     * @return the number of removed nodes
     */
    public int removeLabelNode(String labelName) {
        if (!allLabelNodes.containsKey(labelName)) {
            return 0;
        }
        LabelNode labelNode = allLabelNodes.get(labelName);

        int result = processNodeIndex(labelNode, IndexingAction.Remove);
        if (result > 0 && rootLabelNodes.contains(labelNode)) {
            rootLabelNodes.remove(labelNode);
        }

        return result;
    }

    private int processNodeIndex(LabelNode node, IndexingAction action) {

        int processedNodes = 0;
        if (node.hasChildren()) {
            Set<LabelNode> childNodes = node.getChildren();
            for (LabelNode child : childNodes) {
                processedNodes += processNodeIndex(child, action);
            }
        }
        if (action == IndexingAction.Add) {
            if (allLabelNodes.containsKey(node.getName())) {
                throw new IllegalArgumentException(
                        String.format("The node '%s' is already in the nodes index.", node.getName()));
            }
            allLabelNodes.put(node.getName(), node);
            processedNodes++;
        } else if (action == IndexingAction.Remove) {
            if (allLabelNodes.containsKey(node.getName())) {
                allLabelNodes.remove(node.getName());
                processedNodes++;
            }
        } else {
            throw new MulanRuntimeException(
                    String.format("Indexing action '%s' is not supported.", action));
        }

        return processedNodes;
    }

    /**
     * Will do initialization and indexing of labels structure
     * Prior motivation of this method to be at disposal to {@link LabelsBuilder}.
     * When instance creation from XML is finished (via reflection) some additional
     * processing is required to have API fully operational
     * (e.g. {@link LabelsMetaDataImpl#getLabelNode(String)}).
     */
    void doReInit() {
        allLabelNodes.clear();
        for (LabelNode node : rootLabelNodes) {
            processNodeIndex(node, IndexingAction.Add);
        }
    }

    private enum IndexingAction {

        Add,
        Remove
    }

	@SuppressWarnings("unchecked")
	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		rootLabelNodes = (Set<LabelNode>)in.readObject();
		doReInit();
		
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeObject(this.rootLabelNodes);
	}
}
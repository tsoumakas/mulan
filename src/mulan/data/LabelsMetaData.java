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
 *    LabelsMetaData.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.util.Set;

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
     * @return <code>true</code> if meta data contains the given label name;
     * 		   <code>false</code> otherwise.
     */
    boolean containsLabel(String labelName);

    /**
     * Determines if there is a hierarchy defined between labels. If not, all labels are
     * represented as root {@link LabelNode} nodes.
     *
     * @return <code>true</code> if there is hierarchy defined between labels;
     * 		   <code>false</code> otherwise.
     */
    boolean isHierarchy();

    /**
     * Gets the total number of {@link LabelNode} nodes.
     * @return total number of {@link LabelNode} nodes
     */
    int getNumLabels();

    /**
     * Returns a deep copy of the {@link LabelsMetaDataImpl} instance.
     * @return a deep copy of the {@link LabelsMetaDataImpl} instance
     */
    LabelsMetaData clone();
}
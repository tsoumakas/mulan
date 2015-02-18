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
package mulan.classifier.meta;

import java.util.ArrayList;
import java.util.HashSet;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * Class that implements a node for the {@link HMC}
 *
 * @author George Saridis
 * @version 0.2
 */
public class HMCNode extends MultiLabelMetaLearner {

    private String nodeName;
    private HashSet<HMCNode> children = null;
    private Instances header;
    
    /**
     * Returns the label indices
     * 
     * @return label indices
     */
    public int[] getLabelIndices() {
        return labelIndices;
    }

    /**
     * Creates a new instance with the given name and learner
     * 
     * @param name name of the node
     * @param mlc learner
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public HMCNode(String name, MultiLabelLearner mlc) throws Exception {
        super(mlc);
        nodeName = name;
    }

    /**
     * Returns the header information
     * 
     * @return header
     */
    public Instances getHeader() {
        return header;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        header = new Instances(trainingSet.getDataSet(), 0);
        baseLearner.build(trainingSet);
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        return baseLearner.makePrediction(instance);
    }

    /**
     * Checks whether the node has children
     * 
     * @return whether the node has children
     */
    public boolean hasChildren() {
        if (children != null) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Returns the number of children 
     * 
     * @return number of children
     */
    public int getNumChildren() {
        return children.size();
    }

    /**
     * Returns the children of the current node
     * 
     * @return a set of nodes
     */
    public HashSet<HMCNode> getChildren() {
        return children;
    }

    /**
     * Returns a list of all children labels
     * 
     * @return list of children labels
     */
    public ArrayList<String> getChildrenLabels() {
        ArrayList<String> childrenLabels = new ArrayList<String>();
        for (HMCNode child : getChildren()) {
            childrenLabels.add(child.getName());
        }
        return childrenLabels;
    }

    /**
     * Returns all descendant labels
     * 
     * @return a list of descendant labels
     */
    public ArrayList<String> getDescendantLabels() {
        ArrayList<String> descendantLabels = new ArrayList<String>();
        if (getChildren() != null) {
            for (HMCNode child : getChildren()) {
                descendantLabels.add(child.getName());
                descendantLabels.addAll(child.getDescendantLabels());
            }
        }
        return descendantLabels;
    }

    /**
     * Adds a child to the node.
     *
     * @param child - the child that will be added
     */
    public void addChild(HMCNode child) {
        if (children == null) {
            children = new HashSet<HMCNode>();
        }
        children.add(child);
    }

    /**
     * Returns the name of a node
     * 
     * @return name of the node
     */
    public String getName() {
        return nodeName;
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String globalInfo() {
        return "Class implementing a node in the Hierarchy Of Multi-labEl " +
               "leaRners algorithm. For more information, see\n\n"
                + getTechnicalInformation().toString();
    }
}
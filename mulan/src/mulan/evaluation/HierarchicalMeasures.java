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
 *    HierarchicalMeasures.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation;

import java.util.ArrayList;
import java.util.Set;
import mulan.classifier.*;
import mulan.core.data.LabelNode;
import mulan.core.data.LabelsMetaData;

/**
 * The hierarchical loss is a modified version of the Hamming loss that takes
 * into account an existing hierarchical structure of the labels. It examines the predicted
 * labels in a top-down manner according to the hierarchy and whenever the prediction
 * for a label is wrong, the subtree rooted at that node is not considered further in the
 * calculation of the loss.
 *
 * @author George Saridis
 * @version 0.1
 */
public class HierarchicalMeasures {
    private double hierarchicalLoss;
    private static ArrayList<String> allLabels = null;


	protected HierarchicalMeasures(MultiLabelOutput[] output, boolean[][] trueLabels, LabelsMetaData metaData) {
        computeMeasures(output, trueLabels, metaData);
    }


    /**
     * Calculates hierarchical measures
     */
    private void computeMeasures(MultiLabelOutput[] output, boolean[][] trueLabels, LabelsMetaData metaData) {
        allLabels = new ArrayList<String>();
        for (String label : metaData.getLabelNames())
            allLabels.add(label);

        for (int i = 0; i < output.length; i++) 
            calculateHLoss(output[i].getBipartition(), trueLabels[i], metaData.getRootLabels());
        hierarchicalLoss /= output.length;
    }

    /**
     * Recursively calculates the hierarchical loss
     *
     * @param output
     * @param trueLabels
     * @param children
     */
    protected void calculateHLoss(boolean[] output, boolean[] trueLabels, Set<LabelNode> children) {
        for (LabelNode child : children) {
            int labelIndex = allLabels.indexOf(child.getName());
            if (output[labelIndex] && trueLabels[labelIndex]) {
                calculateHLoss(output, trueLabels, child.getChildren());
            } else {
                hierarchicalLoss += 1;
            }
        }
    }

    public double getHierarchicalLoss() {
        return hierarchicalLoss;
    }
}
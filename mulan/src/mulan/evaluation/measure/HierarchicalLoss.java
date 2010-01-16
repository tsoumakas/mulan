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
 *    HierarchicalLoss.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.data.LabelNode;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the Hierarchical loss measure.
 *
 * @author George Saridis
 * @author Grigorios Tsoumakas
 */
public class HierarchicalLoss extends ExampleBasedMeasure {

    private LabelsMetaData metaData;
    private Map<String, Integer> labelPosition;
    private double loss;

    public String getName() {
        return  "Hierarchical Loss";

    }

    public HierarchicalLoss(MultiLabelInstances data) {
        metaData = data.getLabelsMetaData();

        // calculate the position of labels inside a bipartition
        labelPosition = new HashMap<String, Integer>();
        int[] indices = data.getLabelIndices();
        int counter = 0;
        for (int i : indices) {
            labelPosition.put(data.getDataSet().attribute(i).name(), counter);
            counter++;
        }
    }

    @Override
    protected double updateInternal(MultiLabelOutput prediction, boolean[] truth) {
        boolean[] bipartition = prediction.getBipartition();
        if (bipartition == null) {
            throw new ArgumentNullException("Bipartition is null");
        }
        if (bipartition.length != truth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "bipartition and the ground truth array do not match");
        }

        loss = 0;
        calculateHLoss(bipartition, truth, metaData.getRootLabels());

        sum += loss;
        count++;

        return loss;
    }

    /**
     * Recursively calculates the hierarchical loss
     *
     * @param bipartition
     * @param truth
     * @param children
     */
    protected void calculateHLoss(boolean[] bipartition, boolean[] truth, Set<LabelNode> children) {
        for (LabelNode child : children) {
            int labelPos = labelPosition.get(child.getName());
            if (bipartition[labelPos] == truth[labelPos]) {
                calculateHLoss(bipartition, truth, child.getChildren());
            } else {
                loss += 1;
            }
        }
    }

    public double getIdealValue() {
        return 0;
    }
}

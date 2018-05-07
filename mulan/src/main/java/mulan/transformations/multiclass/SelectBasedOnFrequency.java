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
 *    SelectBasedOnFrequency.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.List;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implement the Select-Max and Select-Min transformation methods.
 *
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public class SelectBasedOnFrequency extends MultiClassTransformationBase {

    /** type of frequency */
    private SelectionType type;
    /** occurences of each label */
    private int[] labelOccurance;

    /**
     * Initializes the transformation with a {@link SelectionType}
     *
     * @param type type of frequency-based selection (MIN/MAX)
     */
    public SelectBasedOnFrequency(SelectionType type) {
        this.type = type;
    }

    @Override
    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        // calculate label occurences
        numOfLabels = mlData.getNumLabels();
        Instances data = mlData.getDataSet();
        labelOccurance = new int[numOfLabels];
        labelIndices = mlData.getLabelIndices();
        int numInstances = data.numInstances();
        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numOfLabels; j++) {
                if (data.instance(i).attribute(labelIndices[j]).value((int) data.instance(i).value(labelIndices[j])).equals("1")) {
                    labelOccurance[j]++;
                }
            }
        }
        return super.transformInstances(mlData);
    }

    /**
     * Transforms a multi-label example to a list containing a single-label
     * multi-class example by selecting the most/least frequent label in the 
     * training set
     *
     * @param instance
     * @return
     */
    List<Instance> transformInstance(Instance instance) {
        int value = labelOccurance[0];
        int labelSelected = 0;
        for (int counter = 1; counter < numOfLabels; counter++) {
            if (instance.attribute(labelIndices[counter]).value((int) instance.value(labelIndices[counter])).equals("1")) {
                boolean test = false;
                switch (type) {
                    case MIN:
                        test = labelOccurance[counter] < value ? true : false;
                        break;
                    case MAX:
                        test = labelOccurance[counter] > value ? true : false;
                        break;
                }

                if (test) {
                    value = labelOccurance[counter];
                    labelSelected = counter;
                }
            }
        }

        Instance transformed = null;
        try {
            transformed = RemoveAllLabels.transformInstance(instance, labelIndices);
            transformed.setDataset(null);
            transformed.insertAttributeAt(transformed.numAttributes());
            transformed.setValue(transformed.numAttributes() - 1, labelSelected);
        } catch (Exception ex) {
            Logger.getLogger(Copy.class.getName()).log(Level.SEVERE, null, ex);
        }

        List<Instance> result = new ArrayList<Instance>();
        result.add(transformed);
        return result;
    }
}
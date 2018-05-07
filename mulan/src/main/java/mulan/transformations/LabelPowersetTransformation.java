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
package mulan.transformations;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implement the Label powerset (LP) transformation method
 *
 * @author Stavros Mpakirtzoglou
 * @author Grigorios Tsoumakas
 * @version 2013.4.17
 */
public class LabelPowersetTransformation implements Serializable {

    private Instances transformedFormat;

    /**
     * Returns the format of the transformed instances
     * 
     * @return the format of the transformed instances
     */
    public Instances getTransformedFormat() {
        return transformedFormat;
    }

    /**
     * 
     * @param mlData multi-label data
     * @return the transformed instances
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        Instances data = mlData.getDataSet();
        int numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();

        Instances newData = null;

        // gather distinct label combinations
        HashSet<LabelSet> labelSets = new HashSet<LabelSet>();
        int numInstances = data.numInstances();
        for (int i = 0; i < numInstances; i++) {
            // construct labelset
            double[] dblLabels = new double[numLabels];
            for (int j = 0; j < numLabels; j++) {
                int index = labelIndices[j];
                dblLabels[j] = Double.parseDouble(data.attribute(index).value((int) data.instance(i).value(index)));
            }
            LabelSet labelSet = new LabelSet(dblLabels);

            // add labelset if not already present
            labelSets.add(labelSet);
        }

        // create class attribute
        ArrayList<String> classValues = new ArrayList<String>(labelSets.size());
        for (LabelSet subset : labelSets) {
            classValues.add(subset.toBitString());
        }
        Attribute newClass = new Attribute("LP_Class_" + Integer.toHexString((int) Math.random()*Integer.MAX_VALUE), classValues);

        // remove all labels
        newData = RemoveAllLabels.transformInstances(data, labelIndices);

        // add new class attribute
        newData.insertAttributeAt(newClass, newData.numAttributes());
        newData.setClassIndex(newData.numAttributes() - 1);

        // add class values
        for (int i = 0; i < newData.numInstances(); i++) {
            //System.out.println(newData.instance(i).toString());
            String strClass = "";
            for (int j = 0; j < numLabels; j++) {
                int index = labelIndices[j];
                strClass = strClass + data.attribute(index).value((int) data.instance(i).value(index));
            }
            //System.out.println(strClass);
            newData.instance(i).setClassValue(strClass);
        }
        transformedFormat = new Instances(newData, 0);
        return newData;
    }

    /**
     * 
     * @param instance the instance to be transformed
     * @param labelIndices the labels to remove.
     * @return tranformed instance
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instance transformInstance(Instance instance, int[] labelIndices) throws Exception {
        Instance transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        transformedInstance.setDataset(null);
        transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
        transformedInstance.setDataset(transformedFormat);
        return transformedInstance;
    }
}
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
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implements the PT6 transformation
 *
 * @author Stavros Mpakirtzoglou
 * @author Grigorios Tsoumakas
 * @version 2012.02.02
 */
public class IncludeLabelsTransformation implements Serializable {

    private int[] labelIndices;

    /**
     *
     * @param mlData multi-label data
     * @return transformed instances
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        int numLabels = mlData.getNumLabels();
        labelIndices = mlData.getLabelIndices();

        // remove all labels
        Instances transformed = RemoveAllLabels.transformInstances(mlData);

        // add at the end an attribute with values the label names
        ArrayList<String> labelNames = new ArrayList<String>(numLabels);
        for (int counter = 0; counter < numLabels; counter++) {
            labelNames.add(mlData.getDataSet().attribute(labelIndices[counter]).name());
        }
        Attribute attrLabel = new Attribute("Label", labelNames);
        transformed.insertAttributeAt(attrLabel, transformed.numAttributes());

        // and at the end a binary attribute
        ArrayList<String> binaryValues = new ArrayList<String>(2);
        binaryValues.add("0");
        binaryValues.add("1");
        Attribute classAttr = new Attribute("Class", binaryValues);
        transformed.insertAttributeAt(classAttr, transformed.numAttributes());

        // add instances
        transformed = new Instances(transformed, 0);
        transformed.setClassIndex(transformed.numAttributes() - 1);
        Instances data = mlData.getDataSet();
        for (int instanceIndex = 0; instanceIndex < data.numInstances(); instanceIndex++) {
            for (int labelCounter = 0; labelCounter < numLabels; labelCounter++) {
                Instance temp;
                temp = RemoveAllLabels.transformInstance(data.instance(instanceIndex), labelIndices);
                temp.setDataset(null);
                temp.insertAttributeAt(temp.numAttributes());
                temp.insertAttributeAt(temp.numAttributes());
                temp.setDataset(transformed);
                temp.setValue(temp.numAttributes() - 2, (String) labelNames.get(labelCounter));
                if (data.attribute(labelIndices[labelCounter]).value((int) data.instance(instanceIndex).value(labelIndices[labelCounter])).equals("1")) {
                    temp.setValue(temp.numAttributes() - 1, "1");
                } else {
                    temp.setValue(temp.numAttributes() - 1, "0");
                }
                transformed.add(temp);
            }
        }

        return transformed;
    }

    /**
     * Transform an unlabeled instance to the format expected by
     * the binary classifier
     *
     * @param instance an unlabeled instance
     * @return a transformed unlabeled instance
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instance transformInstance(Instance instance) throws Exception {
        if (labelIndices == null) {
            System.out.println("Label Indices not set!!");
            return null;
        }
        Instance transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        transformedInstance.setDataset(null);
        transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
        transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
        return transformedInstance;
    }
}

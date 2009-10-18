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
 *    PT6Transformation.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.transformations;

import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Stavros Mpakirtzoglou
 * @author Grigorios Tsoumakas
 */
public class PT6Transformation {

    private int[] labelIndices;

    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        int numLabels = mlData.getNumLabels();
        labelIndices = mlData.getLabelIndices();

        // remove all labels
        Instances transformed = RemoveAllLabels.transformInstances(mlData);

        // add at the end an attribute with values the label names
        FastVector labelNames = new FastVector(numLabels);
        for (int counter = 0; counter < numLabels; counter++) {
            labelNames.addElement(mlData.getDataSet().attribute(labelIndices[counter]).name());
        }
        Attribute attrLabel = new Attribute("Label", labelNames);
        transformed.insertAttributeAt(attrLabel, transformed.numAttributes());

        // and at the end a binary attribute
        FastVector binaryValues = new FastVector(2);
        binaryValues.addElement("0");
        binaryValues.addElement("1");
        Attribute classAttr = new Attribute("Class", binaryValues);
        transformed.insertAttributeAt(classAttr, transformed.numAttributes());

        // add instances
        transformed = new Instances(transformed, 0);
        transformed.setClassIndex(transformed.numAttributes() - 1);
        Instances data = mlData.getDataSet();
        for (int instanceIndex = 0; instanceIndex < data.numInstances(); instanceIndex++) {
            for (int labelCounter = 0; labelCounter < numLabels; labelCounter++) {
                Instance temp = new Instance(data.instance(instanceIndex));
                temp.setDataset(data);
                temp = RemoveAllLabels.transformInstance(temp, labelIndices);
                temp.setDataset(null);
                temp.insertAttributeAt(temp.numAttributes());
                temp.insertAttributeAt(temp.numAttributes());
                temp.setDataset(transformed);
                temp.setValue(temp.numAttributes() - 2, (String) labelNames.elementAt(labelCounter));
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

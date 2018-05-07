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
 *    MultiClassTransformationBase.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import mulan.data.MultiLabelInstances;
import mulan.transformations.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The base class for multi-class transformation methods. It provides initial 
 * implementation of the {@link MultiClassTransformation} interface. All
 * implementations of transformation methods should reuse this base class.
 *  
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public abstract class MultiClassTransformationBase implements Serializable, MultiClassTransformation {

    /** the number of labels */
    protected int numOfLabels;
    /** the array with the label indices */
    protected int[] labelIndices;

    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        labelIndices = mlData.getLabelIndices();
        numOfLabels = mlData.getNumLabels();
        Instances data = mlData.getDataSet();

        Instances transformed = new Instances(mlData.getDataSet(), 0);

        // delete all labels
        transformed = RemoveAllLabels.transformInstances(transformed, labelIndices);

        // add single label attribute
        ArrayList<String> classValues = new ArrayList<String>(numOfLabels);
        for (int x = 0; x < numOfLabels; x++) {
            classValues.add("Class" + (x + 1));
        }
        Attribute newClass = new Attribute("Class", classValues);
        transformed.insertAttributeAt(newClass, transformed.numAttributes());
        transformed.setClassIndex(transformed.numAttributes() - 1);

        for (int instanceIndex = 0; instanceIndex < data.numInstances(); instanceIndex++) {
            //System.out.println(data.instance(instanceIndex).toString());
            List<Instance> result = transformInstance(data.instance(instanceIndex));
            for (Instance instance : result) {
                //System.out.println(instance.toString());
                transformed.add(instance);
                //System.out.println(transformed.instance(transformed.numInstances()-1));
            }
        }
        return transformed;
    }

    abstract List<Instance> transformInstance(Instance instance);
}
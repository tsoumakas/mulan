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
 *    Copy.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.transformations.RemoveAllLabels;
import weka.core.Instance;

/**
 * Class that implement the Copy transformation method
 *
 * @author Stavros Bakirtzoglou
 * @author Grigorios Tsoumakas
 * @version 2012.02.02
 */
public class Copy extends MultiClassTransformationBase {

    /**
     * Transforms a multi-label instance to a list of single-label instances,
     * one for each of the labels that annotate the instance, by copying the
     * feature vector
     *
     * @param instance a multi-label instance
     * @return a list with the transformed single-label instances
     */
    List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        for (int counter = 0; counter < numOfLabels; counter++) {
            if (instance.attribute(labelIndices[counter]).value((int) instance.value(labelIndices[counter])).equals("1")) {
                Instance transformed = null;
                try {
                    transformed = RemoveAllLabels.transformInstance(instance, labelIndices);
                    transformed.setDataset(null);
                    transformed.insertAttributeAt(transformed.numAttributes());
                    transformed.setValue(transformed.numAttributes() - 1, counter);
                } catch (Exception ex) {
                    Logger.getLogger(Copy.class.getName()).log(Level.SEVERE, null, ex);
                }
                result.add(transformed);
            }
        }
        return result;
    }
}
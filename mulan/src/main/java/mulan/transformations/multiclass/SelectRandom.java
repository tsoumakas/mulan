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
 *    SelectRandom.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import java.util.List;
import weka.core.Instance;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.transformations.RemoveAllLabels;

/**
 * Class that implement the Select-Random transformation method
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public class SelectRandom extends MultiClassTransformationBase {

    /**
     * Transforms a multi-label example to a list containing a single-label
     * multi-class example by randomly selecting one of the labels
     * 
     * @param instance the multi-label example
     * @return the list with the single-label multi-class example
     */
    List<Instance> transformInstance(Instance instance) {
        ArrayList<Integer> labels = new ArrayList<Integer>();
        for (int counter = 0; counter < numOfLabels; counter++) {
            if (instance.attribute(labelIndices[counter]).value((int) instance.value(labelIndices[counter])).equals("1")) {
                labels.add(counter);
            }
        }

        int randomLabel = labels.get((int) (Math.random() * labels.size()));

        Instance transformed = null;
        try {
            transformed = RemoveAllLabels.transformInstance(instance, labelIndices);
            transformed.setDataset(null);
            transformed.insertAttributeAt(transformed.numAttributes());
            transformed.setValue(transformed.numAttributes() - 1, randomLabel);
        } catch (Exception ex) {
            Logger.getLogger(Copy.class.getName()).log(Level.SEVERE, null, ex);
        }

        List<Instance> result = new ArrayList<Instance>();
        result.add(transformed);
        return result;
    }
}
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
 *    Ignore.java
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
 * Class that implement the Ignore transformation method
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public class Ignore extends MultiClassTransformationBase {

    /**
     * Transforms a multi-label example with a single annotation to a
     * single-label example and ignores multi-label example with more
     * annotations
     *
     * @param instance a multi-label example
     * @return a list that is either empty or contains the transformed
     * single-label example
     */
    List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        int indexOfSingleLabel = -1;
        int counter = 0;
        for (int labelCounter = 0; labelCounter < numOfLabels; labelCounter++) {
            int index = labelIndices[labelCounter];
            if (instance.attribute(index).value((int) instance.value(index)).equals("1")) {
                counter++;
                indexOfSingleLabel = labelCounter;
            }
            if (counter > 1) {
                break;
            }
        }
        if (counter > 1 || counter == 0) {
            return result;
        }

        Instance transformedInstance;
        try {
            transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
            transformedInstance.setDataset(null);
            transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
            transformedInstance.setValue(transformedInstance.numAttributes() - 1, indexOfSingleLabel);
            result.add(transformedInstance);
        } catch (Exception ex) {
            Logger.getLogger(Ignore.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }
}
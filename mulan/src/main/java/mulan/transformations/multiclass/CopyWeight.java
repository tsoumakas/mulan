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
 *    CopyWeight.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import java.util.List;
import weka.core.Instance;

/**
 * Class that implement the Copy-Weight transformation method
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public class CopyWeight extends Copy {

    /**
     * Transforms a multi-label instance to a list of single-label instances,
     * one for each of the labels that annotate the instance, by copying the
     * feature vector and attaching a weight equal to 1/(list size).
     *
     * @param instance a multi-label instance
     * @return a list with the transformed single-label instances
     */
    @Override
    List<Instance> transformInstance(Instance instance) {
        List<Instance> copy = super.transformInstance(instance);
        for (Instance anInstance : copy) {
            anInstance.setWeight(1.0 / copy.size());
        }
        return copy;
    }
}
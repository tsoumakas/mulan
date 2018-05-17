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
 *    DataUtils.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import mulan.core.MulanRuntimeException;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;

/**
 * Utility class for data related manipulation functions.
 * 
 * @author Jozef Vilcek
 */
public class DataUtils {

    /**
     * Creates a new {@link Instance}. The actual type is determined based on passed instance object.
     * @param typeProvider the instance from which type for new instance is determined
     * @param weight the weight of a new instance
     * @param attrValues attribute values for a new instance
     * @return A new {@link Instance}.
     */
    public static Instance createInstance(Instance typeProvider, double weight, double[] attrValues) {
        if (typeProvider instanceof SparseInstance) {
            return new SparseInstance(weight, attrValues);
        } else if (typeProvider instanceof DenseInstance) {
            return new DenseInstance(weight, attrValues);
        } else {
            throw new MulanRuntimeException(
                    String.format("Can not create a new Instance from supplied type '%s'.",
                    typeProvider.getClass().getName()));
        }

    }

    /**
     * Creates a new {@link Instance}. The actual type is determined based on passed instance object.
     * @param typeProvider the instance from which type for new instance is determined
     * @param numAttributes number of attributes for new instance
     * @return A new {@link Instance}.
     */
    public static Instance createInstance(Instance typeProvider, int numAttributes) {
        if (typeProvider instanceof SparseInstance) {
            return new SparseInstance(numAttributes);
        } else if (typeProvider instanceof DenseInstance) {
            return new DenseInstance(numAttributes);
        } else {
            throw new MulanRuntimeException(
                    String.format("Can not create a new Instance from supplied type '%s'.",
                    typeProvider.getClass().getName()));
        }
    }
}
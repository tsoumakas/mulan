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
 *    MultiClassTransformation.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.transformations.multiclass;

import mulan.data.MultiLabelInstances;
import weka.core.Instances;

/**
 * The interface for single-label multi-class transformations.
 * 
 * @author Stavros Bakirtzoglou
 * @version 2012.02.02
 */
public interface MultiClassTransformation {

    /**
     * Transforms a multi-label dataset to a multi-class single label dataset
     * 
     * @param dataSet a multi-label dataset
     * @return a single-label multi-class dataset
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public Instances transformInstances(MultiLabelInstances dataSet) throws Exception;
}
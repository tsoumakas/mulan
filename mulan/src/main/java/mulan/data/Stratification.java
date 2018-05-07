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
 *    Stratification.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

/**
 * Interface for multi-label stratification methods
 * 
 * @author Grigorios Tsoumakas
 * @version 2012.05.08
 */
public interface Stratification {
    
    /**
     * Creates a number of folds via stratified sampling
     * 
     * @param data a multi-label dataset
     * @param folds the number of folds to sample
     * @return an array of multi-label datasets, one for each fold
     */
    public MultiLabelInstances[] stratify(MultiLabelInstances data, int folds);
}

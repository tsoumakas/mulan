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
 *    LabelPairsDependenceIdentifier.java
 */
package mulan.data;

/**
 *  An interface for various types of dependency identification between pairs of labels.
 * .
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public interface  LabelPairsDependenceIdentifier {

    /**
     *  Calculates dependence level between each pair of labels in the given multilabel data set
     *
     * @param mlInstances multilabel data set
     * @return an array of label pairs sorted in descending order of pairs' dependence score
     */
    public LabelsPair[] calculateDependence(MultiLabelInstances mlInstances);

    /**
     * Returns a critical value
     * @return critical value
     */
    public double getCriticalValue();

}
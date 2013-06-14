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
package mulan.evaluation.measure;

/**
 * Base implementation of the label-based macro/micro precision measures.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public abstract class LabelBasedPrecision extends LabelBasedBipartitionMeasureBase {

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public LabelBasedPrecision(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

}
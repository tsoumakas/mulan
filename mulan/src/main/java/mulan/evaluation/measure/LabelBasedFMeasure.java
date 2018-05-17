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
 * Base implementation of the label-based macro/micro f-measures.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.31
 */
public abstract class LabelBasedFMeasure extends LabelBasedBipartitionMeasureBase {

    /**
     * the parameter for combining precision and recall
     */
    protected final double beta;

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public LabelBasedFMeasure(int numOfLabels) {
        this(numOfLabels, 1);
    }

    /**
     * Constructs a new object with given number of labels and beta parameter
     *
     * @param numOfLabels the number of labels
     * @param beta the beta parameter
     */
    public LabelBasedFMeasure(int numOfLabels, double beta) {
        super(numOfLabels);
        this.beta = beta;
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

}
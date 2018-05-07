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

import weka.core.Utils;

/**
 * Implementation of the micro-averaged precision measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.29
 */
public class MicroFMeasure extends LabelBasedFMeasure {

    /**
     * Constructs a new object with given number of labels and beta=1
     *
     * @param numOfLabels the number of labels
     */
    public MicroFMeasure(int numOfLabels) {
        this(numOfLabels, 1);
    }

    /**
     * Full constructor
     *
     * @param numOfLabels the number of labels
     * @param beta the beta parameter
     */
    public MicroFMeasure(int numOfLabels, double beta) {
        super(numOfLabels, beta);
    }

    @Override
    public double getValue() {
        double tp = Utils.sum(truePositives);
        double fp = Utils.sum(falsePositives);
        double fn = Utils.sum(falseNegatives);
        return InformationRetrievalMeasures.fMeasure(tp, fp, fn, beta);
    }

    @Override
    public String getName() {
        return "Micro-averaged F-Measure";
    }
}
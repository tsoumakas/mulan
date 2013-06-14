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
 * Implementation of GMAP (Geometric Mean Average Precision)
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2010.12.10
 */
public class GeometricMeanAveragePrecision extends MeanAveragePrecision {

    /**
     * Creates a new instance of this class
     * 
     * @param numOfLabels the number of labels
     */
    public GeometricMeanAveragePrecision(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public String getName() {
        return "Geometric Mean Average Precision";
    }

    @Override
    public double getValue() {
        double product = 1;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            product = product * getValue(labelIndex);
        }
        return Math.pow(product, 1.0 / numOfLabels);
    }

    @Override
    public double getIdealValue() {
        return 1;
    }
}
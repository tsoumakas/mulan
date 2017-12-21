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
 * Implementation of the example-based recall measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.29
 */
public class ExampleBasedRecall extends ExampleBasedBipartitionMeasureBase {

    @Override
    public String getName() {
        return "Example-Based Recall";
    }

    @Override
    protected void updateBipartition(boolean[] bipartition, boolean[] truth) {
        double tp = 0;
        double fp = 0;
        double fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                if (truth[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (truth[i]) {
                    fn++;
                }
            }
        }
        sum += InformationRetrievalMeasures.recall(tp, fp, fn);
        count++;
    }

    @Override
    public double getIdealValue() {
        return 1;
    }
}
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
package mulan.evaluation.loss;

/**
 * Interfance for bipartition loss functions
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.01
 */
public interface BipartitionLossFunction extends MultiLabelLossFunction {

    /**
     * Computes the bipartition loss function
     *
     * @param bipartition the biprtition of the learner for an example
     * @param groundTruth the ground truth of the example
     * @return the value of the loss function
     */
    public double computeLoss(boolean[] bipartition, boolean[] groundTruth);
}
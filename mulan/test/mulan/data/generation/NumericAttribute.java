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
package mulan.data.generation;

public class NumericAttribute extends Attribute{
	private final double min;
	private final double max;
	
	public NumericAttribute(String name) {
		this(name, -1, 1);
	}
	
	NumericAttribute(String name, double min, double max){
		super(name, AttributeType.Numeric);
		this.min = min;
		this.max = max;
	}
	
	public double getMin(){
		return min;
	}
	
	public double getMax(){
		return max;
	}
}

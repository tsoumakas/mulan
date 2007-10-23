import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Iterator;
import java.util.Vector;


public class OptionRange implements Iterable<String>
{
	
	private Vector<String> combinations;
	
	public OptionRange(String option)
	{
		combinations = new Vector<String>(2);
		combinations.add(option);
		combinations.add("");
	}
	
	public OptionRange(String option, String[] values)
	{
		combinations = new Vector<String>(values.length);
		for(String v : values) combinations.add(option.concat(" ").concat(v));
	}
	
	public OptionRange(String[] values)
	{
		combinations = new Vector<String>(values.length);
		for(String v : values) combinations.add(v);
	}
	
	
	public OptionRange(String option, double start, double step, int steps)
	{
		DecimalFormat fmt = new DecimalFormat("#0.###");
		DecimalFormatSymbols symbols = new DecimalFormatSymbols();
		symbols.setDecimalSeparator('.');
		fmt.setDecimalFormatSymbols(symbols);
		
		combinations = new Vector<String>(steps);
		combinations.add(option.concat(" ").concat(fmt.format(start)));
		for(int i = 0; i < steps; i++)
			combinations.add(option.concat(" ").concat(fmt.format(start+=step)));
	}

	public OptionRange(String option, int start, int step, int steps)
	{
		combinations = new Vector<String>(steps);
		combinations.add(option.concat(" ").concat(Integer.toString(start)));
		for(int i = 0; i < steps; i++)
			combinations.add(option.concat(" ").concat(Integer.toString(start+=step)));
	}
	
	
	public int size()
	{
		return combinations.size();
	}
	
	public OptionRange combine(OptionRange other)
	{
		int newSize = this.size() * other.size();
		Vector<String> combined = new Vector<String>(newSize);
		for(String s : this)
		{
			for(String t : other )
			{
				combined.add(s.concat(" ").concat(t).trim());
			}
		}
		combinations = combined;
		return this;
	}

	private class OptionRangeIterator implements Iterator<String>
	{
		private int current;
		private OptionRange range;
		
		public OptionRangeIterator(OptionRange range)
		{
			this.range = range;	
		}


		public boolean hasNext()
		{
			return current < range.combinations.size();
		}

		public String next()
		{
			return range.combinations.get(current++);
		}

		public void remove()
		{
			combinations.remove(current);
		}
		
	}
	
	@SuppressWarnings("unchecked")
	public Iterator iterator()
	{
		return new OptionRangeIterator(this);
	}
	


}

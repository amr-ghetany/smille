// A Demo Machine Learning project with smily on Titanic Dataset.
// Note: this demo is for demonstration puposes and does not necessarily produce accurate results 


package omar.smile;

import java.util.List;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.measure.NominalScale;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.plot.swing.Histogram;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;

import com.ctc.wstx.util.StringUtil;

public class TitanicSmileDemo {
    String fullDataPath = "src/main/resources/data/titanic.csv";

    public static void main(String[] args) throws IOException {
        TitanicSmileDemo sd = new TitanicSmileDemo ();
        PassengerDAOCSV pDAO = new PassengerDAOCSV ();
        
        //Load Test and train data
        DataFrame fullData = pDAO.readCSV(sd.fullDataPath);
        DataFrame trainData = fullData.slice(0,891);
        DataFrame testData = fullData.slice(892,1309);

        System.out.println (fullData.structure ());
        System.in.read();
        System.out.println (fullData.summary ());
        System.in.read();
        
        // Clean Data
        trainData = prepareData(trainData);
        testData = prepareData(testData);

        System.out.println ("=======Start of Explaratory Data Analysis==============");
        try {
            eda (fullData);
        } catch (InterruptedException | InvocationTargetException e) {
            e.printStackTrace ();
        }
        
        // Train Model
        RandomForest model = RandomForest.fit(Formula.lhs("Survived"), trainData);
        System.out.println("feature importance:");
        System.out.println(Arrays.toString(model.importance()));
        System.out.println(model.metrics ());
        //Validate Model
        int[][] results = model.test(testData);
        RandomForest model1= model.prune (testData);
        model1.importance ();
        model1.metrics ();
    }

    public static int[] encodeCategory(DataFrame df, String columnName) {
        String[] values = df.stringVector (columnName).distinct ().toArray (new String[]{});
        int[] encodedValues = df.stringVector (columnName).factorize (new NominalScale (values)).toIntArray ();
        return encodedValues;
    }

    public static DataFrame prepareData(DataFrame df){
        df = df.merge (IntVector.of ("Gender", encodeCategory (df, "Sex")));
        df = df.merge (IntVector.of ("PClassValues", df.column("Pclass").toIntArray()));
        System.out.println(df.structure());

        df = df.omitNullRows ();
        df = df.drop("Pclass");
        df = df.drop ("Name");
        df= df.drop("Sex"); 
        
        return df;
    }
    
    private static void eda(DataFrame titanic) throws InterruptedException, InvocationTargetException {
        titanic.summary ();
        DataFrame titanicSurvived = DataFrame.of (titanic.stream ().filter (t -> t.get ("Survived").equals (1)));
        DataFrame titanicNotSurvived = DataFrame.of (titanic.stream ().filter (t -> t.get ("Survived").equals (0)));
        titanicNotSurvived.omitNullRows ().summary ();
        titanicSurvived = titanicSurvived.omitNullRows ();
        titanicSurvived.summary ();
        int size = titanicSurvived.size ();
        System.out.println (size);
        Double averageAge = titanicSurvived.stream ()
                .mapToDouble (t -> t.isNullAt ("Age") ? 0.0 : t.getDouble ("Age"))
                .average ()
                .orElse (0);
        System.out.println (averageAge.intValue ());
        Map map = titanicSurvived.stream ()
                .collect (Collectors.groupingBy (t -> Double.valueOf (t.getDouble ("Age")).intValue (), Collectors.counting ()));

        double[] breaks = ((Collection<Integer>) map.keySet ())
                .stream ()
                .mapToDouble (l -> Double.valueOf (l))
                .toArray ();

        int[] valuesInt = ((Collection<Long>) map.values ())
                .stream ().mapToInt (i -> i.intValue ())
                .toArray ();

        Histogram.of (titanicSurvived.doubleVector ("Age").toDoubleArray (), 15, false)
                .canvas ().setAxisLabels ("Age", "Count")
                .setTitle ("Age frequencies among surviving passengers")
                .window ();
        System.out.println (titanicSurvived.schema ());
        //////////////////////////////////////////////////////////////////////////

    }
}

package WEKA;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;

import java.util.Random;

public class kNN_EKORKETA {

    private static Instances data;

    public static void main(String[] args) throws Exception {
        int kEstimatu = Integer.parseInt(args[1]);
        datuak_Kargatu(args[0]);
        K_Parametroaren_Ekorketa(kEstimatu);
        parametroGuztienEkorketa(kEstimatu);
    }

    private static void parametroGuztienEkorketa(int kEstimatu) throws Exception {
        LinearNNSearch linearNNSearch = new LinearNNSearch();
        NormalizableDistance[] distantziak = new NormalizableDistance[3];
        distantziak[0] = new EuclideanDistance();
        distantziak[1] = new ManhattanDistance();
        distantziak[2] = new MinkowskiDistance();

        double maxMeasure = 0.0;
        int k = 1;
        int w = 0;
        IBk knn = new IBk();
        for(int i=1; i<=kEstimatu; i++){
            knn.setKNN(i);
            for(int j=0; j<distantziak.length; j++){
                linearNNSearch.setDistanceFunction(distantziak[j]);
                knn.setNearestNeighbourSearchAlgorithm(linearNNSearch);

                Evaluation evaluation = new Evaluation(data);
                evaluation.crossValidateModel(knn,data,10,new Random(1));
                if(evaluation.weightedFMeasure() > maxMeasure){
                    maxMeasure = evaluation.weightedFMeasure();
                    k = i;
                    w = j;
                }
            }
        }
        System.out.println("K hoberena -> " + k);
    }

    private static void K_Parametroaren_Ekorketa(int kEstimatu) throws Exception {
        double maxMeasure = 0.0;
        int k = 1;
        for(int i=1; i<=kEstimatu; i++) {
            IBk knn = new IBk();
            knn.buildClassifier(data);
            knn.setKNN(i);

            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(knn,data,10,new Random(1));
            System.out.println(evaluation.weightedFMeasure());
            if(evaluation.weightedFMeasure() > maxMeasure) {
                maxMeasure = evaluation.weightedFMeasure();
                k=i;
            }
        }
        System.out.println("\nEMAITZA -> " + maxMeasure + " ETA K -> " + k);
    }

    private static void datuak_Kargatu(String path) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        data = source.getDataSet();
        if(data.classIndex() == -1) data.setClassIndex(data.numAttributes()-1);
    }
}

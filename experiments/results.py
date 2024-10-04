import matplotlib.pyplot as plt
import json


def g_prototypical_networks():
    speakers = ['ovidio', 'jose', 'armando', 's1', 's2', 's3', 's4','carlos']
    vals = [50.0, 100.0, 54.14, 40.0, 50.0, 61.54, 55.56, 40.91]

    plt.figure(figsize=(8, 6))
    plt.bar(speakers, vals, color='skyblue')

    plt.xlabel('Hablante')
    plt.ylabel('Precision (%)')
    plt.title('Prototypical Networks')

    plt.ylim(0, 100)

    plt.show()


def g_matching_networks():
    speakers2 = ['unknown', 'ovidio', 'jose', 'armando', 's1', 's2', 's3', 's4', 'carlos']
    vals2 = [81.82, 50.0, 75, 85.71, 100, 100, 23.08, 33.56, 45.46]

    plt.figure(figsize=(8, 6))
    plt.bar(speakers2, vals2, color='skyblue')

    plt.xlabel('Hablante')
    plt.ylabel('Precision (%)')
    plt.title('Matching Networks')

    plt.ylim(0, 100)

    plt.show()

g_matching_networks()
g_prototypical_networks()
def transcription_precision():
    results = json.load(open('transcription_results.json', 'r'))
    t = 0
    for res in results:
        t += res['similarity']
    print(t / len(results))


def transcription_precision_w():
    results = json.load(open('transcription_results_w.json', 'r'))
    t = 0
    for res in results:
        t += res['similarity']
    print(t / len(results))


transcription_precision()
transcription_precision_w()

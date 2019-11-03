import hljs from 'highlight.js/lib/highlight';
import hljsLanguages from './hljsLanguages';
import HTTP, { defaultHttpClient } from './http';
import Messages from './messages.vue';

hljsLanguages.forEach((languageName) => {
  /* eslint-disable import/no-dynamic-require, global-require */
  const languageModule = require(`highlight.js/lib/languages/${languageName}`);
  /* eslint-enable import/no-dynamic-require, global-require */
  hljs.registerLanguage(languageName, languageModule);
});

export default {
  components: { Messages },

  data: () => ({
    file: '',
    messages: [],
    format: 'json',
    isLoading: false,
    isCloudUploadActive: false,
    canUploadFromCloud: false,
  }),

  mounted() {
    hljs.initHighlighting();
  },

  created() {
    defaultHttpClient.get('/v1/features').then((response) => {
      this.canUploadFromCloud = response.data.cloud_upload;
    });
  },

  computed: {
    projectId() {
      return window.location.pathname.split('/')[2];
    },

    postUploadUrl() {
      return window.location.pathname.split('/').slice(0, -1).join('/');
    },

    cloudUploadUrl() {
      return '/cloud-storage'
        + `?project_id=${this.projectId}`
        + `&upload_format=${this.format}`
        + `&next=${encodeURIComponent('about:blank')}`;
    },
  },

  methods: {
    cloudUpload() {
      const iframeUrl = this.$refs.cloudUploadPane.contentWindow.location.href;
      if (iframeUrl.indexOf('/v1/cloud-upload') > -1) {
        this.isCloudUploadActive = false;
        this.$nextTick(() => {
          window.location.href = this.postUploadUrl;
        });
      }
    },

    upload() {
      this.isLoading = true;
      this.file = this.$refs.file.files[0];
      const formData = new FormData();
      formData.append('file', this.file);
      formData.append('format', this.format);
      HTTP.post('docs/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then((response) => {
          console.log(response); // eslint-disable-line no-console
          this.messages = [];
          window.location = this.postUploadUrl;
        })
        .catch((error) => {
          this.isLoading = false;
          this.handleError(error);
        });
    },

    //  upload_3(file) {
    //   this.isLoading = true;
    //   this.file = this.$refs.file.files[0];
    //   const formData = new FormData();
    //   formData.append('file', this.file);
    //   formData.append('format', this.format);
    //   HTTP.post('docs/upload',
    //     formData,
    //     {
    //       headers: {
    //         'Content-Type': 'multipart/form-data',
    //       },
    //     })
    //     .then((response) => {
    //       console.log(response); // eslint-disable-line no-console
    //       this.messages = [];
    //       window.location = this.postUploadUrl;
    //     })
    //     .catch((error) => {
    //       this.isLoading = false;
    //       this.handleError(error);
    //     });
    // },

    upload_2() {
      this.isLoading = true;
      const headers = {};
      if (this.format === 'csv') {
        headers.Accept = 'text/csv; charset=utf-8';
        headers['Content-Type'] = 'text/csv; charset=utf-8';
      } else {
        headers.Accept = 'application/json';
        headers['Content-Type'] = 'application/json';
      }
      HTTP({
        url: 'docs/download',
        method: 'GET',
        responseType: 'blob',
        params: {
          q: this.format,
        },
        headers,
      }).then((response) => {
        var myObj = {"text": "LANGSA - Polres Langsa, Selasa (12/2) dini menangkap Husaini (55) warga Desa Pulo Baro Tangse, Pidie, ketika berupaya menyelundupkan 12 ekor trenggiling. <EOL> Sampai tadi malam pria tersebut bersama barang bukti 12 ekor binatang yang dilindungi itu masih diamankan di Mapolres setempat. <EOL> Kapolres Langsa, AKBP Hariadi SIK, melalui Kasat Reskrim AKP Muhammad Firdaus, kepada Serambi mengatakan, keberhasilan penangkapan penyelundup 12 ekor trenggiling itu setelah ada laporan dari masyarakat. <EOL> Dalam laporan masyarakat itu disebutkan bahwa seorang pria dari Tangse Pidie sedang membawa 12 ekor trenggiling ke Medan Husaini, sedang membawa sebanyak 12 ekor trenggiling tujuan ke Medan dengan menumpangi bus Kurnia. <EOL> Atas informasi tersebut, kata AKP Muhammad Firdaus, sejumlah anggota Satuan Reskrim Polres Langsa, Selasa (12/2) sekitar pukul 03.00 WIB menjelang pagi, memberhentikan bus yang sudah ketahui nomor polisinya, tepat di depan Mapolres Langsa. Katanya, setelah digeledah di bagian bagasi bus, ditemukan 12 ekor trenggiling yang berada dalam satu karung goni besar. <EOL> Kasat Reskrim menambahkan, dari 12 ekor trenggiling itu, satu diantaranya suda mati. Setelah meenurunkan barang bukti (BB) trenggiling itu, selanjutnya petugas mencari tahu pemilik barang terlarang itu.  <EOL> \u201cSaat itu juga tersangka Husaini dan BB sebanyak 12 ekor trenggiling langsung diamankan ke Mapolres untuk mempertanggung jawabkan perbuatannnya,\u201dujarnya. Tersangka mengaku trenggiling itu akan dibawa ke Medan, dan di sana suda ada yang menunggu untuk membeli binatang dilindungi tersebut. <EOL> Menurut AKP Firdaus, atas perbuatannya itu tersangka Husaini dikenakan Pasal 21 ayat 1 dan 2 Jo Pasal 40 ayat 2 dan 4 Undang-Undang Nomor 5 Tahun 1990, tentang sumber daya hayati dan ekosistem dengan ancaman hukuman penjara lima tahun. Sementara itu sebanyak 12 ekor trenggiling dilindungi tersebut akan diserahkan kepada Badan Konservasi Sumber Daya Alam (BKSDA) Aceh.(c42) <EOS>", "labels": [[0, 6, "LOCATION"], [16, 23, "LOCATION"], [31, 37, "DATE"], [43, 52, "CRIME_TYPE"], [95, 101, "LOCATION"], [118, 132, "CRIME_TYPE"], [141, 153, "WILDLIFE"], [221, 229, "WILDLIFE"], [303, 310, "LOCATION"], [425, 436, "CRIME_TYPE"], [445, 456, "WILDLIFE"], [604, 615, "WILDLIFE"], [625, 633, "LOCATION"], [666, 677, "WILDLIFE"], [773, 781, "LOCATION"], [821, 828, "LOCATION"], [836, 842, "DATE"], [857, 862, "DATE"], [927, 937, "WILDLIFE"], [962, 969, "LOCATION"], [1037, 1048, "WILDLIFE"], [1137, 1148, "WILDLIFE"], [1220, 1231, "WILDLIFE"], [1364, 1375, "WILDLIFE"], [1413, 1427, "CRIME_TYPE"], [1479, 1490, "WILDLIFE"], [1500, 1506, "CRIME_TYPE"], [1510, 1516, "LOCATION"], [1558, 1565, "CRIME_TYPE"], [1869, 1880, "WILDLIFE"]]};
        const test_2 = JSON.stringify(myObj);
        const blob = new Blob([test_2], { type: 'application/json' });

        const file = new File([ blob ], 'FileName.json');

        console.log('========================================>');
        console.log(file);

        const formData = new FormData();
        formData.append('file', file, 'FileName.json');
        formData.append('deal_id', dealId);


        console.log(formData);

        // console.log(test);
        // console.log('=============================================')
        // console.log(response.data)

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'file.' + this.format); // or any other extension
        document.body.appendChild(link);
        this.isLoading = false;
        link.click();
      }).catch((error) => {

        this.isLoading = false;
        this.handleError(error);
      });
    },

    handleError(error) {
      console.log('3')
      const problems = Array.isArray(error.response.data)
        ? error.response.data
        : [error.response.data];

      problems.forEach((problem) => {
        if ('detail' in problem) {
          this.messages.push(problem.detail);
        } else if ('text' in problem) {
          this.messages = problem.text;
        }
      });
    },

    download() {
      this.isLoading = true;
      const headers = {};
      if (this.format === 'csv') {
        headers.Accept = 'text/csv; charset=utf-8';
        headers['Content-Type'] = 'text/csv; charset=utf-8';
      } else {
        headers.Accept = 'application/json';
        headers['Content-Type'] = 'application/json';
      }
      HTTP({
        url: 'docs/download',
        method: 'GET',
        responseType: 'blob',
        params: {
          q: this.format,
        },
        headers,
      }).then((response) => {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'file.' + this.format); // or any other extension
        document.body.appendChild(link);
        this.isLoading = false;
        link.click();
      }).catch((error) => {
        this.isLoading = false;
        this.handleError(error);
      });
    },
  },
};

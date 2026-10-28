# Крупные данные, не опубликованные в GitHub

Обновлено 28.10.2026. По просьбе пользователя эти файлы исключены из
публикуемого снимка. Все они сохранены в исходной локальной папке проекта;
изменения индекса Git не удаляют их с диска. Правила в `.gitignore`
предотвращают случайное повторное добавление.

Итого: **217 файлов, 2,784,182,701 байт**
(**2.593 ГиБ**).

Здесь находятся полная RF-траектория A за 50 Ts, её блоки продолжения,
плотные короткие траектории и промежуточные P2/P4-массивы.
Основные отчёты, графики, сетки и выбранные поля/медленные данные
опубликованы; точный состав описан в [GITHUB_DATA.md](GITHUB_DATA.md).

**Сохраните отдельную резервную копию этих файлов.** Новый клон и
`git pull` их не восстанавливают. Для полного повторного анализа и
продолжения исходного запуска нужны соответствующие оригиналы.
Сохранённые научные манифесты не изменены: проверки полного исследования
в сокращённом клоне могут сообщать об отсутствующих входах.

Список ниже описывает отсутствие файлов в опубликованном дереве Git.
Часть объектов могла попасть в хранилище LFS при прерванной попытке;
это не делает их частью доступного снимка и не заменяет резервную копию.

## Полный список

Пути относительно корня репозитория. SHA-256 вычислены по исходным байтам.

| Путь | Размер, байт | SHA-256 |
|---|---:|---|
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/common_samples.npz` | 1232639 | `34015be312e3c2dc7314a7817737722f53e2d0f5dc91e7539fe2ff111df645eb` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/sector_solution.npz` | 10529781 | `da90df98f9db31b5b7d84397bc9605bc6a4b940d8b30a8f9acb7d19cd8815a3d` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_coarse_extended.npz` | 50781856 | `35bed8f4e1b6bef7b668b75a5008c65e54501bcd62cd6e1582ca7fda854d02c8` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_coarse_pilot.npz` | 16303394 | `ded699f72213bc147f40b63a5b979c11faadab83f1d66f56dc52b4226e8bfde3` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_fine_extended.npz` | 49763265 | `f65d977be45d5ee38bcf4f429806a7021c3633895bdd5f4fdfb19ca640ae9a83` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_fine_pilot.npz` | 15977707 | `c3ab6efb63f4ebd687c6b90dbd1efe4711cd9774da675636b72fabaa267db9e2` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_matched_phase0_extended.npz` | 49470448 | `4074c4b97603143c1807e734f09878f2ea06ebcc9e4e4f3f61a2c70c35187aa0` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_matched_phase0_pilot.npz` | 15832377 | `817ae2e5f649ee44f935ed1cac6ad7b93c7cbebe7e9395bb5745cd3cc5c33838` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/B_matched_phase1_pilot.npz` | 15790980 | `1808a283ee925824f3533f63e704a4b8ac5496d31b79c10aff796404c2c75fed` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/C_coarse_extended.npz` | 26716371 | `e357f9c3ce9237ba1eded60a47f64f2598464ed2d608683843d4ce8321f8a91b` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/C_coarse_pilot.npz` | 8603966 | `13bb99b6a3f010cbaa3374e696d6c5986a8e3f7835ea67ebe96639890624788e` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/C_fine_extended.npz` | 25682722 | `ca17a9ee15165b1017115fb3c05f1cd528cd134790f84d0114aa3e6fe6bccc73` |
| `outputs/abc_validation/direct_fem_validation/v1/h35_P4/slow/C_fine_pilot.npz` | 8273860 | `8788828136db0042308b21aa900d9f45ba12a75466f54ca31f2db0024720387f` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/common_samples.npz` | 1232904 | `eb8d97dda473ad3fe73580ef75b5d2940961d6483ab222c5786b392bec048735` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/sector_solution.npz` | 5244446 | `336882933c476bc7ecd2fd97bdc060991ddda5b6d685a3c0889e5ee36b493872` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_coarse_extended.npz` | 50820211 | `d3b408abf6f115336bcfaaf8e7404d75df86fa003913db3deebc917b856c2939` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_coarse_pilot.npz` | 16326683 | `973b14bae6c2b635912dc582b1eaafbe5fbf8e42f366c4422b8859a7e154f19b` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_fine_extended.npz` | 49799646 | `87550bba45c51454a55142e9b95e1ac5035ae6c9a01c19ee332e66f2cd0a32be` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_fine_pilot.npz` | 16001670 | `ff4eac6d08ad088e28ba1056a09def4a45a93b15fbf2e3da3e9273e34024904b` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_matched_phase0_extended.npz` | 49468277 | `ca1e685cdb706ecbcfaf49eb9eafe40450bea73a0406c91717c5b51dea045db2` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_matched_phase0_pilot.npz` | 15832223 | `a8698834f7232c52c55c0a6c95bb30bc00e0373d4dee7d52ad4fbb326f6aa90d` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/B_matched_phase1_pilot.npz` | 15789047 | `17f43f542d17f49131c82e3b5df04372d0246684ab3749c98b57485cd265d155` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/C_coarse_extended.npz` | 26718879 | `29cc20fa2fb94aa475c5a0beac5476c7e969512e2ca5c687d7391363c65ce6f2` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/C_coarse_pilot.npz` | 8604284 | `047d7d5f20644c44a4ef2a975c6876188a9a278a0493297e36b043eaa34d7b23` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/C_fine_extended.npz` | 25683833 | `d2da1d8b11224e5da792171691c6580ff8ab6240efcb0c6919398f8a606e890c` |
| `outputs/abc_validation/direct_fem_validation/v1/h50_P4/slow/C_fine_pilot.npz` | 8274472 | `c5c5adec68255fff72ef75dd4632aef41825af7f0f777cfe9dc897a02fc43d28` |
| `outputs/abc_validation/direct_fem_validation/v1/h70_P2/common_samples.npz` | 1252410 | `3656e9e980c4bb0c489c0fd173a776d1aaa12d8cfde3af2558525fae16cfcc86` |
| `outputs/abc_validation/direct_fem_validation/v1/h70_P2/sector_solution.npz` | 1272350 | `31d916830a0e769bbf0f43daab53cb0a0ddd71bb97361908bf6ff9718a1d5eda` |
| `outputs/abc_validation/direct_fem_validation/v1/h70_P4/common_samples.npz` | 1234609 | `4faf270ef8b938d3ac06115e87e7cb2df5e3898112e4023dbface4c7f3b9737e` |
| `outputs/abc_validation/direct_fem_validation/v1/h70_P4/sector_solution.npz` | 2757111 | `45d17278106c5b2ee0a449d696463d507c64925a80f8fc78c471a0e17c6dc6c1` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_fit_h35_phase0_fine_pilot.npz` | 15655373 | `072741310604472c9f66eb5b37dcae148ef1fd99e1d9cc02a6de877f1296f40d` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_fit_h35_phase1_fine_pilot.npz` | 15654424 | `d8475328869a3e7f6186197f0ebd8ded5fd0e53b1100e995487465994cdb073a` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_h50_phase0_coarse_pilot.npz` | 16061720 | `1f8946dc198b2a3b43f88ab253474483607e3033603415f490c4054a9f683fae` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_h50_phase0_fine_pilot.npz` | 16062212 | `ce8bc0dfbbda2858602916b9f5b4558c3bf4f86f7e7a08f5a2f7447f2dffc3a4` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_h50_phase1_coarse_pilot.npz` | 16060937 | `91311d791b38171eef8111b96fe3f21a425859587bf1b37b000685664d07c667` |
| `outputs/abc_validation/direct_fem_validation/v1/runs/A_h50_phase1_fine_pilot.npz` | 16060928 | `84ceef9f7f592d329226dc06dedadd8c692ef500bd3b3c5b85e5722bacc67957` |
| `outputs/abc_validation/direct_fem_validation/v2/A35_phase0_extended_poincare.npz` | 1691037 | `6580a906d999b56a716f813e7973f3ab34877ad431558847101f28d006d97e1a` |
| `outputs/abc_validation/direct_fem_validation/v2/A35_phase0_extended_slow.npz` | 33184393 | `4ea074a16cdc36e396b415809a4b4c1a30ed32934c0b3c9fd8171540673d2c54` |
| `outputs/abc_validation/direct_fem_validation/v2/A35_phase0_pilot_slow.npz` | 10574834 | `f3e6b3ca16cc5f21180dbd10b22d464cb29980c8da43cdb70fdf3ad941449230` |
| `outputs/abc_validation/direct_fem_validation/v2/A35_phase1_pilot_slow.npz` | 10573951 | `8faff2374262f5a8c238116d8ce643b27c9d052b6119370b491a5f7a6f36ea1b` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_coarse_extended.npz` | 50781856 | `35bed8f4e1b6bef7b668b75a5008c65e54501bcd62cd6e1582ca7fda854d02c8` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_coarse_pilot.npz` | 16303394 | `ded699f72213bc147f40b63a5b979c11faadab83f1d66f56dc52b4226e8bfde3` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_fine_extended.npz` | 49763265 | `f65d977be45d5ee38bcf4f429806a7021c3633895bdd5f4fdfb19ca640ae9a83` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_fine_pilot.npz` | 15977707 | `c3ab6efb63f4ebd687c6b90dbd1efe4711cd9774da675636b72fabaa267db9e2` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_matched_phase0_extended.npz` | 49470448 | `4074c4b97603143c1807e734f09878f2ea06ebcc9e4e4f3f61a2c70c35187aa0` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_matched_phase0_pilot.npz` | 15832377 | `817ae2e5f649ee44f935ed1cac6ad7b93c7cbebe7e9395bb5745cd3cc5c33838` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/B_matched_phase1_pilot.npz` | 15790980 | `1808a283ee925824f3533f63e704a4b8ac5496d31b79c10aff796404c2c75fed` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/C_coarse_extended.npz` | 26716371 | `e357f9c3ce9237ba1eded60a47f64f2598464ed2d608683843d4ce8321f8a91b` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/C_coarse_pilot.npz` | 8603966 | `13bb99b6a3f010cbaa3374e696d6c5986a8e3f7835ea67ebe96639890624788e` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/C_fine_extended.npz` | 25682722 | `ca17a9ee15165b1017115fb3c05f1cd528cd134790f84d0114aa3e6fe6bccc73` |
| `outputs/abc_validation/direct_fem_validation/v2/h35_P4/slow/C_fine_pilot.npz` | 8273860 | `8788828136db0042308b21aa900d9f45ba12a75466f54ca31f2db0024720387f` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_coarse_extended.npz` | 50820211 | `d3b408abf6f115336bcfaaf8e7404d75df86fa003913db3deebc917b856c2939` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_coarse_pilot.npz` | 16326683 | `973b14bae6c2b635912dc582b1eaafbe5fbf8e42f366c4422b8859a7e154f19b` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_fine_extended.npz` | 49799646 | `87550bba45c51454a55142e9b95e1ac5035ae6c9a01c19ee332e66f2cd0a32be` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_fine_pilot.npz` | 16001670 | `ff4eac6d08ad088e28ba1056a09def4a45a93b15fbf2e3da3e9273e34024904b` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_matched_phase0_extended.npz` | 49468277 | `ca1e685cdb706ecbcfaf49eb9eafe40450bea73a0406c91717c5b51dea045db2` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_matched_phase0_pilot.npz` | 15832223 | `a8698834f7232c52c55c0a6c95bb30bc00e0373d4dee7d52ad4fbb326f6aa90d` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/B_matched_phase1_pilot.npz` | 15789047 | `17f43f542d17f49131c82e3b5df04372d0246684ab3749c98b57485cd265d155` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/C_coarse_extended.npz` | 26718879 | `29cc20fa2fb94aa475c5a0beac5476c7e969512e2ca5c687d7391363c65ce6f2` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/C_coarse_pilot.npz` | 8604284 | `047d7d5f20644c44a4ef2a975c6876188a9a278a0493297e36b043eaa34d7b23` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/C_fine_extended.npz` | 25683833 | `d2da1d8b11224e5da792171691c6580ff8ab6240efcb0c6919398f8a606e890c` |
| `outputs/abc_validation/direct_fem_validation/v2/h50_P4/slow/C_fine_pilot.npz` | 8274472 | `c5c5adec68255fff72ef75dd4632aef41825af7f0f777cfe9dc897a02fc43d28` |
| `outputs/abc_validation/direct_fem_validation/v2/h70_P2/common_samples.npz` | 1252410 | `3656e9e980c4bb0c489c0fd173a776d1aaa12d8cfde3af2558525fae16cfcc86` |
| `outputs/abc_validation/direct_fem_validation/v2/h70_P2/sector_solution.npz` | 1272350 | `31d916830a0e769bbf0f43daab53cb0a0ddd71bb97361908bf6ff9718a1d5eda` |
| `outputs/abc_validation/direct_fem_validation/v2/h70_P4/common_samples.npz` | 1234609 | `4faf270ef8b938d3ac06115e87e7cb2df5e3898112e4023dbface4c7f3b9737e` |
| `outputs/abc_validation/direct_fem_validation/v2/h70_P4/sector_solution.npz` | 2757111 | `45d17278106c5b2ee0a449d696463d507c64925a80f8fc78c471a0e17c6dc6c1` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_fit_h35_phase0_fine_extended.npz` | 49145945 | `5500455f3bc5229ac4434ded3e2db2c326eb282666683925bf8a86a1badace18` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_fit_h35_phase0_fine_pilot.npz` | 15655373 | `072741310604472c9f66eb5b37dcae148ef1fd99e1d9cc02a6de877f1296f40d` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_fit_h35_phase1_fine_pilot.npz` | 15654424 | `d8475328869a3e7f6186197f0ebd8ded5fd0e53b1100e995487465994cdb073a` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_coarse_extended.npz` | 50391268 | `2f24409937b04483e00e3af82d02e22d64b731a29644194abcbc2203f4229f18` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_coarse_pilot.npz` | 16063543 | `58ff49f96d885c0118b04e7be99b607c6bdd015e043c195d8ba9aa222778be67` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_fine_extended.npz` | 50391104 | `f5a4287af425c9574ed30da6a1f31670e43b1ea47e74af4f3531fa0b2cd70321` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_fine_pilot.npz` | 16063966 | `5a233e79d631e746cbbcfd80cdf1e3cae6de8d491b2fd57ad1b43120397c7d6d` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_radau_pilot.npz` | 16064050 | `5c01ed86221026d4bc5241aec8895241d5779bda550731ed11b915f0f15ada38` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase0_refined_pilot.npz` | 16063621 | `27f4e8a4de45f05721abd0c118c80210aa77c8392f17ad247c952fc8d6004f35` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase1_coarse_pilot.npz` | 16062284 | `51335affc7b4e12bd1087c1c80016165fdefcb5f448f30b3fcfa25628a753b19` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h35_phase1_fine_pilot.npz` | 16062502 | `6de021885b99f45368bbd3aebbd55b3476bb38088e92a7342ecd74b2e6d85145` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase0_coarse_extended.npz` | 50388596 | `ae38aa7a6e68f0f7445b07514a7aed415a86b7a602d932d4fd4538605875c94a` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase0_coarse_pilot.npz` | 16061720 | `1f8946dc198b2a3b43f88ab253474483607e3033603415f490c4054a9f683fae` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase0_fine_extended.npz` | 50388734 | `5a0f1fe575395c96d42e27ed8af883a13a45fd8debdaa73dd116edead2d75945` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase0_fine_pilot.npz` | 16062212 | `ce8bc0dfbbda2858602916b9f5b4558c3bf4f86f7e7a08f5a2f7447f2dffc3a4` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase1_coarse_pilot.npz` | 16060937 | `91311d791b38171eef8111b96fe3f21a425859587bf1b37b000685664d07c667` |
| `outputs/abc_validation/direct_fem_validation/v2/runs/A_h50_phase1_fine_pilot.npz` | 16060928 | `84ceef9f7f592d329226dc06dedadd8c692ef500bd3b3c5b85e5722bacc67957` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/B35_coarse.npz` | 8229939 | `e15b8b4898f64367dcf914cfd973aec70b22f8d6fece45cc7a1653db86c6f5ae` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/B35_matched_fine.npz` | 7954146 | `d763467e5f919e684bd0f17527c84f849138d08e95a80ff2cad6a42f4b62add6` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/B70_coarse.npz` | 8226001 | `8b4669a3e10513bbda801e10b420727a55780a0f2e5556a72761332322b8680c` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/B70_fine.npz` | 7978357 | `18428887c055c0e57f3742ea3d21d0e3dcfb3ed8dc5d81d2b44b33e2e0f9e988` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/C35_coarse.npz` | 4571660 | `d27bc1fa239f832696b53a84755fda5a70216a6b9688b90fefe467c53f962b34` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/C70_coarse.npz` | 4571804 | `87684229c24f10e8d9b56d64314b252d8722e8267f112e7c304f37c94f797c4d` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/analysis/v1/references/C70_fine.npz` | 4322497 | `15be9d713cceca445eb45c59342d630d126f33c21d920e09bdd5a092bd8bc2ba` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00000.npz` | 4230154 | `60fea44a224fba1273ff80903ce220a069193ff20af3a1d844f053665ed662af` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00001.npz` | 4321145 | `1d43fecedd9617a563e84908b68e11e319255694fc7dd4167990069902d75426` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00002.npz` | 4291734 | `cbf8cdfdd89220513d910184fd7e0eec297c120c85cab193df5410e28cf8eed1` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00003.npz` | 4281687 | `488b468fa308a49b42e86ed79e60ffd554e52ce3954fbbde11457d8f65aac91c` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00004.npz` | 4274315 | `0a39aa242713db0fcfb003ee3d7269f46ef3361d3e67ffe2273ca34521a88cfe` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00005.npz` | 4281812 | `7a8fec0ec1f5a7d439cc72f15225c162d613c07d78d950dee0aa6a621ea70d2d` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00006.npz` | 4265008 | `590be6bbc2447045f84f76251e5d9ddaff511af2ef726b38f263eeaaf65f5da9` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00007.npz` | 4264187 | `cf05fb67f3984ef198e54c660e0a48bbb72c37fddbeceb0bc5ae3b148fb95648` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00008.npz` | 4257561 | `7932a6bd607cfc254f32a73bf5ef1c8dda9f8ddc6fe01bdf769a873f20b19176` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00009.npz` | 4256901 | `be622404923069ff8dd9ebb7b27c589cad8ed54fb3367f5e882d12767c429444` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00010.npz` | 4245988 | `44ea6ce693ef90e100c12a379687bb43e0835bab53544a2556de32c18bd6f4aa` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00011.npz` | 4248216 | `19eeb27656c6223d9aaf8b069d814f8fcf8be30aefb99bdbe87477e369cf9821` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00012.npz` | 4244375 | `3c08fb342099b0d30becaa940dd2ec4e0a6a824ac584adc74482664eea48313d` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00013.npz` | 4242230 | `ad1955c47424bf6fa632e22d15bb40ca651c0e4ce7e0c3b4cf55de0b7cd69278` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00014.npz` | 4238281 | `26661284a1b76945af473b838341d0d53b6b680f1f778deb24076404dc6de20c` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00015.npz` | 4239406 | `fa4e2519281a45e2efef77547eef570529edc40fb87b83c392e8d71798a010c8` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00016.npz` | 4241179 | `c2b6ff1f8403977938319daab71e46f0edccdd52e96502d0492e46d9f048e730` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00017.npz` | 4230932 | `865e8ee3f4f31c7dc825c13dbfdb363648f6cfebc5bff82b1b7f7de75253ccb1` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00018.npz` | 4233886 | `04cb28d7870b0441198f83c902184537f86502f0e73c4bdea77b3c68ca740991` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00019.npz` | 4232965 | `416da0af292bd01fa5bff102bcc6c38b4880e8130382cf6db9f0db38e8901abe` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00020.npz` | 4234180 | `3f35dabbcd8a921f4256a10eff77df2216a051348f366ea0044d1faa9b0d140b` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00021.npz` | 4225689 | `4078fdb44fec8bf30877e53dbd7a76af8abc18184af1fb31c27c95c19db775b6` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00022.npz` | 4229369 | `ff8f1380b5c3b7e5aa0a13b1bfec139958cc8c62db9a173e45e7832f3df4f01f` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00023.npz` | 4229785 | `33ed36871b4747fe9f1ce2d069dc44ebc8a6d08025fd64ca91ba38d4a1c5d4b1` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00024.npz` | 4231292 | `8475df08038025bd26100da15714044823655e088b10bc005f92a21d0600a75b` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00025.npz` | 4221276 | `6f1557c110f55099f7339cc511c791f150426824f7e92d6c599bd931150694de` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00026.npz` | 4225583 | `5b68eeb1a288432da8fc6edd642e825d430861d915b3801fb0620bdf404cf2ee` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00027.npz` | 4227443 | `cf6b22816116d9cb1d3245a6fd9cb1519a7183e356f55a60266d0c9a900e4f67` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00028.npz` | 4223368 | `7a7d82ea973dff0c6da3cc41b175b7570a423f62eeec7591a78b33083f3499fe` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00029.npz` | 4221202 | `33586e2eafd346ec638bf60f9d7cae3acfca1e1261dd999d9a8de24bd82f5c81` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00030.npz` | 4222086 | `38700da830b86d2068fae7ed06de81b495441c4f7dc99b429b3432f42986b9be` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00031.npz` | 4230155 | `673e5518ea094235bad5f1ed5383920887a00e629c9df829aa31217c905cbd25` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00032.npz` | 4217088 | `7ecfed16627c8f05662e2d9216f12b0a1fc313d0df1dbe28367391551579466e` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00033.npz` | 4219231 | `31102a2a8b570728ee90a14f22122ceed42cd75c5502a67de251f166d74778fc` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00034.npz` | 4222036 | `049086f306731cb865be65a3e870c4ecd3977c4a8cd409756f84b064a5068b46` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00035.npz` | 4225775 | `0dff3a3e2939cf2005adb2c199a142546fa6c65d8c8ef4f027b37119beb9b89e` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00036.npz` | 4214741 | `c56f60239147af4b43c6a8a25e93c2acd16e413f437283109ab6b1728347d974` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00037.npz` | 4219562 | `2d39402cfb208523da31ce0cd9c068451a770d0be98c88b009fea34ca3f6a2e2` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00038.npz` | 4218651 | `49551b3736c268b3d9c2f6092ad48ba330c4911addf9e3a5cc209afa022e2bb3` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00039.npz` | 4221061 | `e8be5da6b106f2a44c293e103f3757e5dd67bfbea6c7c4430f312acc52e91bf3` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00040.npz` | 4212546 | `83731ad4c3bf6ccd2ac8f4d4e60f0e65429c31b7b1fe97d26069383b775d7840` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00041.npz` | 4215241 | `64f04e8cb5108f621e6e6eb2db2f87912fee7d0595785b42de6a18d9ca3097b6` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00042.npz` | 4225147 | `dac405f5d464bac85cd43f69a18e09cb4ccb60cb12a0ee3f28a758b6501eb0e4` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00043.npz` | 4213189 | `6d818f1bd817476e9b54130454dc0db53da6cbecb2042db2990a6739a439b712` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00044.npz` | 4213068 | `eb020b347a8db7deaca85a7568d268b1a6a61c63f56967ea2bdac072421c76ac` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00045.npz` | 4216495 | `7a5ebd2b57cfb54be7327de4f63cb74a610394330e77b481e88ba1fd7c31c3d0` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00046.npz` | 4223455 | `4627c96f34bb7564adcc10538b60b12c25e7f839eb5335a94956aaec09a7e6e1` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00047.npz` | 4209474 | `3ccc617f26e98f1f1e0d03beefe55104f7a113238013d177d936ef861e4bb1fe` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00048.npz` | 4214170 | `bae2389b49a57cc5ebb0b161e53f12a3a0940d3d501c71e3a3fc5327eefed198` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00049.npz` | 4215746 | `47f3027bd90a31c229e693885822f7836e7264edcc08a6647beb54080721f8a2` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00050.npz` | 4218508 | `9e426f9261824b0d30a8691f25329d1f3cff361181ef787fd7de447362025dba` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00051.npz` | 4207123 | `f1d1a9a6dc03ca4a445a09d77fd652f3dbdccd584e743630544243ba1ddec432` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00052.npz` | 4211923 | `a5e2671ea0299bb582d0c011eb5770da33a0c2d85a3aa28c52ce2327d5f13385` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00053.npz` | 4216892 | `aa09b925a4afd5922171ac90443a5d8b4eca7aefb93b11b30bac629b60371595` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00054.npz` | 4215633 | `0f348884996afd0f53be8fe63d46ecf71055d27c5425e337c44fd3c57a47aa7e` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00055.npz` | 4208383 | `37bc2a5ef8df4b42033e8824530b39442099939f122ea77d8e67bb8ac200ed2f` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00056.npz` | 4213569 | `84312358fead72b504f4783c6956932a17bb1fda95305d7323efd967e99d0c0d` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/checkpoints/part_00057.npz` | 3098127 | `5c06b3104005149dfd46ef5c665d63500b7654d1c94181d6b4eedb53c9c1b0b5` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero/trajectory.npz` | 243962706 | `436afc5d6b9afa9a1833c6576531492b8bb94a9e6e45ebccc60f7da0edea5de9` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_smoke_RF64/checkpoints/part_00000.npz` | 73020 | `233fa1dda94249cfadc214725f701ce984e7c2efa0331cc9b234a2c12335f52b` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_smoke_RF64/checkpoints/part_00001.npz` | 75593 | `7d469cf09782927fbeeebe6a137a09c7b876b5a874ad458c3bd06f84dabdf52c` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_smoke_RF64_1a3a29b0/checkpoints/part_00000.npz` | 73020 | `233fa1dda94249cfadc214725f701ce984e7c2efa0331cc9b234a2c12335f52b` |
| `outputs/abc_validation/long_A/v1/A_P4_h35_phase0_fine_smoke_RF64_1a3a29b0/checkpoints/part_00001.npz` | 75593 | `7d469cf09782927fbeeebe6a137a09c7b876b5a874ad458c3bd06f84dabdf52c` |
| `outputs/abc_validation/raw_edge_audit/v1/A_phase0_fine_pilot.npz` | 15993833 | `8593fac4c9c67ea9452e66f0a15eb21d40f3c447248f99664e8161323be44dcd` |
| `outputs/abc_validation/raw_edge_audit/v2/A_phase0_fine_pilot.npz` | 15993833 | `8593fac4c9c67ea9452e66f0a15eb21d40f3c447248f99664e8161323be44dcd` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop100_common.npz` | 1087288 | `aa324ca10a2d8354c267b34b243f14b1be621be5f26658dfb4977ee5c90bc2e0` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop200_common.npz` | 1087409 | `5801af558778d1b7a2bbfe3d14cb1976757de39669780aa417fb4340416c2ff3` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop200_plus.npz` | 1087271 | `18335953ed3a12a70dfb08692ccffbb380e2ef2a5a6b7ceb5e756c36ba0949ea` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop400_common.npz` | 1087374 | `db2e153e104e6b8e65b51a0ab7be47d6f8c02a0447c4ad29eb4c5143a58b9247` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop400_minus.npz` | 1087349 | `fbca18bc470baa56c0984176a35c1633f92ee7490ea8457ca9dfe8992c98354c` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_dop400_plus.npz` | 1087390 | `de9b0e460f80b6d15d67c6c4b9d76dd16c00f0c4a7678e8a7658d615628c9788` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_radau100_common.npz` | 1087282 | `4000982779a1df93c5c3c166cefcb756eaa6e25afeedb5676d9af8d6281293e4` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/early_radau50_common.npz` | 1087326 | `98284e11c756f2d3a9a3aeb4eebd1c7746405ff1444e62c526b770fffc039259` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/historical_growth.npz` | 8450610 | `f32e9ce1254c0a8722b662a4a81ddb2ef5c721f66c86610d6f6cc1163b674b2c` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/long_dop200_common.npz` | 4396818 | `1c32e4c8b560095d5a3ad83adef8a064e7b49e4f270af4b802c42250246a2b7a` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/long_dop200_plus.npz` | 4396542 | `b1cf7df0c7724558ef3752845b21f9e415b08b7d2c5a9fceff2fea26207eaa2d` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/long_dop400_common.npz` | 4396718 | `19008a80e7f6ed335e312f0cb2bbc1b4731f84186b8cf0bce1abe7c896b806de` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/long_dop400_minus.npz` | 4395498 | `c14236097a327d5b9bbc4382221f9015e3d8138709580664251f31d5294c18ba` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/long_dop400_plus.npz` | 4395975 | `3b5ed40c654004cd8f4d820cd7814a57f3e1c22f21427a78d79ae617e5a42c6f` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop100_common.npz` | 1110364 | `e04d5acb0edc5b39f2d0969db2e99f2f8ff547e812cbbb5768bdcc0a296f455b` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop200_common.npz` | 1110285 | `b384c8de1af1b2131c46e2bcab2add204a3e341889e407e3a50b76422e9e65a5` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop200_plus.npz` | 1110293 | `8dceda1bef7c5b78d9273b47051bb7103cbc0ac224007b975c40028fdef518b3` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop400_common.npz` | 1110300 | `123c09afc7edfa52ae402b22f908f1cb9fc596f76c6d4280cf8f12ab33a13fce` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop400_minus.npz` | 1110484 | `87402f78e909e34392c73ae60e4ef68813fc37a9ab2296cfa12efc9b9e36cbce` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_dop400_plus.npz` | 1110350 | `b1035576c36c286287698a5b6c98cd978e91dad873581ab816dd21db20d6a767` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_radau100_common.npz` | 1110307 | `b8b80907e935796983bc3abc3ebf3c1a77fb2cc0139d6e0c6a32c6f4d1f7eb56` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/onset_radau50_common.npz` | 1110293 | `46d8fa3e652830f73a20bb44aa04fc6a92092611f4ef8434bc7f52f7f78e8013` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop100_common.npz` | 1107175 | `fefc9fdf81b32cfef0d641a504bf57a65409eec9951f613e58981bbdb22281f1` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop200_common.npz` | 1107070 | `d1b70ec3418f0449bf0c05170aaa8f6450f187fa281cc09dac3d162f67c62ecd` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop200_plus.npz` | 1107401 | `f24b03eb1672ace4ed20caabeea847481c1c88a6bbbff5b1b91b7428b9002295` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop400_common.npz` | 1107090 | `3d2f9ff1ceb50c11a9c98aeff46f803cde47288664f451b257869d364acd296e` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop400_minus.npz` | 1107054 | `d22f41bdaa053b1a4cc81d52751e1e9b6ac5f9ed6e56a77b3338f7a51cb858c9` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_dop400_plus.npz` | 1107358 | `4911e754932eadb59987cd4ffb43fb74f6b0dbecd1780f4640a10ab4b24ebacf` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_radau100_common.npz` | 1107178 | `5d9170c51ea3c4000aed55d11f6754652457ddee826911da1c549feca0285a26` |
| `outputs/abc_validation/raw_restart_diagnosis/v1/rapid_radau50_common.npz` | 1107267 | `0952a5e7710abb03ef54555a86217e855a536f2c916865fb7a4b2a9e43dba882` |
| `outputs/abc_validation/raw_spatial_diagnosis/v1/h70/common_samples.npz` | 6620238 | `e56aa383dc36bb965569f9e3e7c876300f9c4c0b93a61f522e1b962ca5212f14` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/A_smooth_coarse.npz` | 16057894 | `b82b1bfe52957fefdcd56532a0551239c7f699af69302e99062e8764604fd6d9` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/A_smooth_fine.npz` | 15655974 | `d72a9e7855c4be72c160b49eb9b71b9c8b3dba2760310c15547397797869daba` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/B_coarse.npz` | 15670237 | `221ac0d78bb63d38359d1813253cbd98487a8a523362fefbe0931df389635c7b` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/B_fine.npz` | 15671176 | `38673d9e41ba57c4f71891084ae41b4d74a66f98ab07471ec17ff5e50b981c61` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/common_samples.npz` | 6677656 | `bab0a5bedc663d89b80d8b93f6bacd5265c906f4f7d689c01693f7721243703e` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/facet_samples.npz` | 3166903 | `58fe7a550dcf1b0440aeef389ff62a62b2cf0cb8386487bd2ab8a2474f17768e` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/local_raw_dop200.npz` | 1106762 | `e08d80e16946604a77dc540562d2ac5abc6366a6524529ad36a920dcdb31f8ee` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/local_raw_dop400.npz` | 1106734 | `2ef5d9f35d744eaadcfe14e86a3a796d4bf5712b3ad3f2c0daa7ddbc48c027af` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/local_smooth_100.npz` | 1083918 | `4b9ec50071dea4759a33767cc67770f93305ac38ad60fb3d720977333fe90175` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h35/local_smooth_200.npz` | 1083973 | `19e874020559407e449aacafc4b1ae4f418bc4be249e162d0bda538f27a69768` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/A_smooth_coarse.npz` | 16057882 | `25d0da03c01bfea2d84ce05d2c76207e5e6356d7419cd24fcef86fa31d208a98` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/A_smooth_fine.npz` | 15655502 | `afdaaaac32af674159238a84ed029e65360508d2f31ac383bdf06ea4dbf57df4` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/B_coarse.npz` | 15498086 | `0b2902482152de708f7cd3872f16235a5f4821f91cfd581246b8146fc2aa85e9` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/B_fine.npz` | 15498059 | `2a8f6a003aeedda87657b282e6c883e63e2f9a8e7f88cf1d1b1936898441d1a1` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/common_samples.npz` | 6647699 | `e140b896271a32819bfd6ef311d7f5f123f18378eea1ee61a06937c77dbd1844` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/facet_samples.npz` | 1526505 | `f272c3f094ec3e2c5bf33f7d147526644a2a71fbafe95118b5f9293c2799206a` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/local_raw_dop200.npz` | 1107161 | `e465128be535fc19880afa96d55c0a5d293998a8c8849e052266cff00df565d4` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/local_raw_dop400.npz` | 1107131 | `aaaedeea3f54aff56b0f56e78ec43666e2c3dc1e57159f472909538b0f2e2096` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/local_smooth_100.npz` | 1084036 | `b3560ffd2bcaec964dbfb8e918b9a4e08dc7db810fcccb85aa9c6f60aa8337c5` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h50/local_smooth_200.npz` | 1084115 | `887b9674e0a68acd8ad159bb70923f585d9a7c0718cf910322a2a65691610ee9` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/A_smooth_coarse.npz` | 16058363 | `897b082307b8fbf62fbebee95b2515bcc537b07731868deb06bf2799adc6339a` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/A_smooth_fine.npz` | 15656149 | `12dc79ec64a9e16ee45fbbda55d928c021f359d7b8a564d301c14f32979ef3a1` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/B_coarse.npz` | 15492033 | `57435f09d43446878cd3cf221be578537e45a5dfebb3f4b37bc5e7717895d191` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/B_fine.npz` | 15492881 | `f6ce12d09cd73000331deb58dfa4cb9c5f974c3ea6c79d7d903255901468ab13` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/common_samples.npz` | 6620238 | `e56aa383dc36bb965569f9e3e7c876300f9c4c0b93a61f522e1b962ca5212f14` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/local_raw_dop200.npz` | 1106587 | `57c8b0add71d8b977212099714c173689dfa999263dc81abe362a0163c68cb30` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/local_raw_dop400.npz` | 1106607 | `4610b073af2dbf8126446445ed5fd3650267e781608076eb98962b1b2030537c` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/local_smooth_100.npz` | 1084024 | `9540bf0a696da07de19871300119d2a30636b72700ee0707883bb2b371259c61` |
| `outputs/abc_validation/raw_spatial_diagnosis/v2/h70/local_smooth_200.npz` | 1084025 | `33e6b61d83519aa5bb05117ccbc89e985840dba77294e68df11e0becfca1336c` |
| `outputs/fem_cache/variants/4169ae3c40d920944fc1/p2_basis.npz` | 113542477 | `9020502649750e86fa1b85639b1995ddf76743af5991d4b463b26de902e8173f` |
| `outputs/fem_cache/variants/915052d897401e5b84c6/p2_basis.npz` | 55922243 | `a432a439d5b0d02cdc6547a96d002a7c2fac63059e06b435562df7576b84ca49` |

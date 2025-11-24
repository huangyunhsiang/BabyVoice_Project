                # 提取特徵
                rms = librosa.feature.rms(y=y)
                avg_volume = np.mean(rms)
                
                centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                avg_pitch_feature = np.mean(centroids)
                
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
                bpm = tempo[0]

                # 3. 顯示波形圖
                st.subheader("2. 聲學監控儀表板")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#FF4B4B') 
                ax.set_title(f"Recorded Waveform ({seconds}s)")
                st.pyplot(fig)

                # 顯示數據
                col1, col2, col3 = st.columns(3)
                col1.metric("能量強度 (RMS)", f"{avg_volume:.4f}")
                col2.metric("音高頻率 (Hz)", f"{avg_pitch_feature:.0f}")
                col3.metric("節奏 (BPM)", f"{bpm:.0f}")

                # 4. 邏輯判斷
                predicted_type = "未知"
                urgency_color = "blue"
                
                # 判斷邏輯
                if avg_volume > 0.08 and avg_pitch_feature > 2800:
                    predicted_type = "疼痛 (Pain)"
                    urgency_color = "red"
                elif bpm > 110 and avg_volume > 0.04:
                    predicted_type = "飢餓 (Hunger)"
                    urgency_color = "orange"
                elif avg_volume < 0.03:
                    predicted_type = "疲倦 (Tired)"
                    urgency_color = "blue"
                else:
                    predicted_type = "不適/尋求關注"
                    urgency_color = "green"

                # 5. 顯示結果
                st.subheader("3. 智慧決策建議")
                
                if urgency_color == "red":
                    st.error(f"### 分析結果：{predicted_type}")
                elif urgency_color == "orange":
                    st.warning(f"### 分析結果：{predicted_type}")
                else:
                    st.success(f"### 分析結果：{predicted_type}")

                # SOP 建議
                advice = ""
                if "疼痛" in predicted_type:
                    advice = "🔴 **緊急檢查 SOP：**\n1. 檢查是否有外傷或頭髮纏繞手指。\n2. 量測體溫。\n3. 按壓腹部確認是否脹氣 (腸絞痛)。\n**若安撫無效請立即就醫。**"
                elif "飢餓" in predicted_type:
                    if last_feed < 1.5:
                        advice = "🟡 **判斷建議：**\n距離上一餐時間短，可能是**口慾期討奶嘴**或**需要拍嗝**。"
                    else:
                        advice = "🟢 **判斷建議：**\n生理時鐘與哭聲特徵吻合，**建議立即餵食**。"
                elif "疲倦" in predicted_type:
                    advice = "🔵 **判斷建議：**\n寶寶累了，請減少環境刺激（關燈、白噪音），進行哄睡。"
                else:
                    if is_diaper_clean == "髒/濕":
                        advice = "🟡 **判斷建議：**\n請優先更換尿布。"
                    else:
                        advice = "🟢 **判斷建議：**\n可能是無聊或想要抱抱，建議變換姿勢或對話互動。"

                st.markdown(advice)
                
                # 清除暫存檔
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        except Exception as e:
            st.error(f"錄音或分析失敗：{e}")
            st.warning("請確認您的電腦有接上麥克風，且沒有被其他程式佔用。")
else:
    st.info("等待指令... 請點擊上方紅色按鈕開始錄音。")